# ============================================================================
# FINAL SCRIPT: Equivariance Error Evaluation
# All functions are implemented directly. No dummy classes.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
import csv
import time
import random
import hashlib
import json
import logging
import warnings
from torch.utils.checkpoint import checkpoint
from interpolant_1xbig import EquiJumpDiT, gamma_tau, TimestepEmbedding, ConditioningMLP
from utils import CONFIG_KEYS, parse_config_file, cfg2vec, complex_to_real_channel_stack, real_to_complex_channel_stack
from rotate_alm_helper import healpix_lmax_from_C, rotate_trajectory
from loading_vanilla import apply_healpix_normalization_final, HealpixSequenceDataset, merge_coeffs
import sys

# ============================================================================
# SECTION 2: EQUIVARIANCE EVALUATION SCRIPT
# ============================================================================

model_name = sys.argv[1] 
num_pairs = 500 #set number of pairs evaluated here
#model_name = "big_0rots_very_big"

# --- Silence warnings ---
logging.getLogger('healpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Paths & Constants ---

stats_file = "../sh_norm_stats_correct.pkl"
val_file   = "../test_folders_final.txt"
root_path  = ".."
ckpt_path  = f"../interpolant_results/lipidgen_best_{model_name}.pt"
cfg_stats_path = "../config_norm.pkl"
RESULTS_FILE = f"equivariance_{model_name}.csv"


SDE_STEPS     = 100
LMAX_RADIAL, LMAX_FG, LMAX_BG = 14, 22, 44
C_pos_RADIAL, C_pos_FG, C_pos_BG = hp.Alm.getsize(LMAX_RADIAL), hp.Alm.getsize(LMAX_FG), hp.Alm.getsize(LMAX_BG)
M_RADIAL, M_FG, M_BG = 1, 9, 3

SHAPE_INFO = {
    "radial": {"C_pos": C_pos_RADIAL, "M": M_RADIAL},
    "fg": {"C_pos": C_pos_FG, "M": M_FG},
    "bg": {"C_pos": C_pos_BG, "M": M_BG},
}

# --- Setup Model and Data ---
if not os.path.exists(stats_file): raise FileNotFoundError(f"Missing {stats_file}")
if not os.path.exists(val_file): raise FileNotFoundError(f"Missing {val_file}")
if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Missing {ckpt_path}")
if not os.path.exists(cfg_stats_path): raise FileNotFoundError(f"Missing {cfg_stats_path}")

val_folds = [ln.strip() for ln in open(val_file) if ln.strip()]
ds_val = HealpixSequenceDataset(root_path, val_folds, stats_path=stats_file, cfg_stats_path=cfg_stats_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

temp_traj = ds_val[0]
rad_dim, fg_dim, bg_dim = [temp_traj[k].shape[1] * temp_traj[k].shape[2] for k in ["radial", "fg", "bg"]]
dims = (rad_dim, fg_dim, bg_dim)
D = sum(dims)
P = len(temp_traj["cond"])

net = EquiJumpDiT(D, cond_dim=P, dims=dims).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt.get("model_state", ckpt)
net.load_state_dict(state, strict=True)
net.eval()
print("✅ Plain model loaded")

# --- Evaluation Helper Functions ---

def _random_angles(seed_str: str):
    seed = int(hashlib.sha1(seed_str.encode()).hexdigest(), 16) % 2**32
    rng = random.Random(seed)
    psi, theta, phi = rng.uniform(0, 2*math.pi), math.acos(1 - 2*rng.random()), rng.uniform(0, 2*math.pi)
    return psi, theta, phi

def rotate_flat_coeffs(flat_coeffs_real, angles, dims, shape_info):
    """
    Rotates a flat real-valued SH coefficient vector by adapting to the
    [R_all, I_all] data layout from the provided utils.py.
    """
    device = flat_coeffs_real.device
    rad_flat, fg_flat, bg_flat = torch.split(flat_coeffs_real, dims, dim=0)
    rotated_parts_flat = []

    for part_flat, key in zip([rad_flat, fg_flat, bg_flat], shape_info.keys()):
        info = shape_info[key]
        C_pos, M = info["C_pos"], info["M"]
        
        part_real_2d = part_flat.reshape(C_pos, M * 2)
        part_cplx = real_to_complex_channel_stack(part_real_2d.unsqueeze(0)).squeeze(0)
        
        part_cplx_rotated_np = rotate_trajectory(part_cplx.unsqueeze(0).numpy(), *angles)
        part_cplx_rotated = torch.from_numpy(part_cplx_rotated_np).squeeze(0)
        
        part_real_rotated_2d = complex_to_real_channel_stack(part_cplx_rotated.unsqueeze(0)).squeeze(0)
        rotated_parts_flat.append(part_real_rotated_2d.flatten())
        
    return torch.cat(rotated_parts_flat).to(device)

def sample_next_frame(net, x_t, cond, seed):
    """Encapsulates the SDE sampling loop, setting the seed for reproducibility."""
    torch.manual_seed(seed)
    X_sde = torch.zeros(SDE_STEPS + 1, 1, D, device=device)
    X_sde[0] = x_t
    taus = torch.linspace(0, 1.0, SDE_STEPS, device=device).unsqueeze(-1).unsqueeze(-1)
    
    with torch.no_grad():
        for k, tau_val in enumerate(taus):
            x_tau_k = X_sde[k]
            x_tau3d = x_tau_k.unsqueeze(1)
            z_tau = torch.randn_like(x_tau3d)
            
            gamma = gamma_tau(tau_val)
            b_hat, eta_hat = net(x_t=x_t.unsqueeze(1), x_tau=x_tau3d, cfg=cond, tau=tau_val)
            
            noise_modulator = 1.2 / (gamma + 0.05)
            drift = (b_hat.squeeze(1) - noise_modulator * eta_hat.squeeze(1)) / SDE_STEPS
            diff  = math.sqrt(2.0 / SDE_STEPS) * z_tau.squeeze(1)
            X_sde[k+1] = x_tau_k + drift + diff
            
    return X_sde[-1]

# --- Main Evaluation Loop ---
print("\nStarting equivariance error evaluation...")
all_results = []


for i in range(num_pairs):
    print(f"Processing frame pair {i+1}/{num_pairs}...")
    traj_idx = random.randint(0, len(ds_val) - 1)
    traj = ds_val[traj_idx]
    if traj["radial"].shape[0] < 2: continue
    
    start_frame = random.randint(0, traj["radial"].shape[0] - 2)
    seq = torch.cat([merge_coeffs(traj[k].unsqueeze(0)) for k in ["radial", "fg", "bg"]], 2).to(device)
    cond = traj["cond"].unsqueeze(0).to(device)
    x_t_original = seq[:, start_frame]
    
    rotation_seed_str = f"{traj['folder']}_{i}"
    angles = _random_angles(rotation_seed_str)
    sde_seed = random.randint(0, 2**32 - 1)
    
    # Experiment 1: Rotate Input, then Sample
    x_t_rotated = rotate_flat_coeffs(x_t_original.flatten().cpu(), angles, dims, SHAPE_INFO).to(device)
    predicted_state_1 = sample_next_frame(net, x_t_rotated.unsqueeze(0), cond, sde_seed)
    
    # Experiment 2: Sample, then Rotate Output
    predicted_state_2_real = sample_next_frame(net, x_t_original, cond, sde_seed)
    predicted_state_2_rotated = rotate_flat_coeffs(predicted_state_2_real.flatten().cpu(), angles, dims, SHAPE_INFO).to(device)
    
    # Calculate and store errors
    mae = F.l1_loss(predicted_state_1.flatten(), predicted_state_2_rotated.flatten()).item()
    mse = F.mse_loss(predicted_state_1.flatten(), predicted_state_2_rotated.flatten()).item()
    
    all_results.append({
        "folder": traj["folder"], "frame_index": start_frame, "mae": mae, "mse": mse,
        "psi": angles[0], "theta": angles[1], "phi": angles[2],
    })
    print(f"  ► MAE: {mae:.6f} | MSE: {mse:.6f}")

print("\n✅ Evaluation finished.")

# --- Save Results to CSV and Plot ---
if all_results:
    header = all_results[0].keys()
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ Results successfully saved to {RESULTS_FILE}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Equivariance Error Distribution', fontsize=16)
    ax1.hist([r['mae'] for r in all_results], bins=20, color='skyblue', edgecolor='black')
    ax1.set_title('Mean Absolute Error (MAE)'); ax1.set_xlabel('Error Value (Lower is Better)'); ax1.set_ylabel('Frequency')
    ax2.hist([r['mse'] for r in all_results], bins=20, color='salmon', edgecolor='black')
    ax2.set_title('Mean Squared Error (MSE)'); ax2.set_xlabel('Error Value (Lower is Better)'); ax2.set_ylabel('Frequency')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"equivariance_{model_name}.png")
    print(f"✅ Histogram plot saved to equivariance_error_final.png")
    plt.show()
else:
    print("No results were generated to save or plot.")