# ============================================================================
# Auto-regressive sampler for the real-valued EquiJumpCore
# ============================================================================
import torch, healpy as hp, numpy as np, matplotlib.pyplot as plt, math, pickle, os, csv, time
from loading_vanilla import (HealpixSequenceDataset, collate_complex,
                             merge_coeffs, unnorm_healpix_alms_final)
from utils import real_to_complex_channel_stack, CONFIG_KEYS
from interpolant_1xbig import EquiJumpDiT, gamma_dot
from interpolant_1xbig import gamma_tau
# NEW: Import for metrics calculation
from sklearn.metrics import mean_absolute_error, jaccard_score, f1_score, accuracy_score
import logging
import warnings
logging.getLogger('healpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 0. Paths & constants -------------------------------------------
import sys
model_name = sys.argv[1] 
#model_name = "big_0rots_very_big"
MAX_PREDICTION_HORIZON = 100  # How many frames to predict into the future


stats_file  = "../sh_norm_stats_correct.pkl"
val_file    = "../test_folders_final.txt"
root_path   = ".."
ckpt_path   = f"../interpolant_results/lipidgen_best_{model_name}.pt"
RESULTS_FILE = f"results_{model_name}.csv"

# --- Constants from your original script ---
SDE_STEPS     = 100
LMAX_RADIAL   = 14
LMAX_FG       = 22
LMAX_BG       = 44
NSIDE_RADIAL  = 16
NSIDE_PHASE   = 16
N_MAPS        = 3
N_FG_CLASSES  = 3
eps_const = 1.0

def main():
    # ---------- 1.  Dataset and stats -------------------------------------------
    all_stats = pickle.load(open(stats_file, "rb"))
    val_folds = [ln.strip() for ln in open(val_file) if ln.strip()]
    ds_val    = HealpixSequenceDataset(root_path, val_folds, stats_path=stats_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 2. Build model dimensions (only needs to be done once) --------
    # We can infer dimensions from the first sample
    temp_traj = ds_val[0]
    rad_dim = temp_traj["radial"].shape[1] * temp_traj["radial"].shape[2]
    fg_dim  = temp_traj["fg"].shape[1] * temp_traj["fg"].shape[2]
    bg_dim  = temp_traj["bg"].shape[1] * temp_traj["bg"].shape[2]
    dims    = (rad_dim, fg_dim, bg_dim)
    D       = rad_dim + fg_dim + bg_dim
    P       = len(CONFIG_KEYS)

    # -------- 3. Instantiate *plain* model and load checkpoint ---------------
    ckpt = torch.load(ckpt_path, map_location=device)

    net = EquiJumpDiT(D, cond_dim=P, dims=dims).to(device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in state):
        state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    print("✅ plain model loaded")

    # ==============================================================================
    # 4. NEW: Evaluation Metrics Calculation Functions (REFINED)
    # ==============================================================================

    def calculate_metrics(pred_maps, true_maps):
        """
        Calculates all required metrics for a single predicted frame.
        This version reflects that healpy maps are already 1D arrays.
        """
        metrics = {}

        # Radial MAE (already a 1D array)
        metrics['radial_mae'] = mean_absolute_error(true_maps['radial'], pred_maps['radial'])

        # Phase Map Metrics
        for i in range(1, N_MAPS + 1):
            map_key = f'phase_{i}'
            # The maps are already 1D arrays (flat vectors) of size 3072
            true_map_1d = true_maps[map_key]
            pred_map_1d = pred_maps[map_key]
            
            # Standard metrics
            metrics[f'{map_key}_iou'] = jaccard_score(true_map_1d, pred_map_1d, average='weighted')
            metrics[f'{map_key}_weighted_f1'] = f1_score(true_map_1d, pred_map_1d, average='weighted')
            metrics[f'{map_key}_accuracy'] = accuracy_score(true_map_1d, pred_map_1d)

            # Background accuracy (0 vs. other classes)
            true_binary = (true_map_1d > 0).astype(int)
            pred_binary = (pred_map_1d > 0).astype(int)
            metrics[f'{map_key}_background_accuracy'] = accuracy_score(true_binary, pred_binary)
            
        return metrics

    # ==============================================================================
    # 5. Setup for Reconstruction (Your original code, slightly modified for reuse)
    # ==============================================================================
    # These should match the channel counts in your data
    N_PHASE_MAPS = 3
    N_FG_CLASSES_PER_MAP = 3

    # --- Calculate sizes for BOTH complex (pos-m) and real basis vectors ---
    C_pos_RADIAL = hp.Alm.getsize(LMAX_RADIAL)
    C_pos_FG     = hp.Alm.getsize(LMAX_FG)
    C_pos_BG     = hp.Alm.getsize(LMAX_BG)
    C_real_RADIAL = (LMAX_RADIAL + 1)**2 + (LMAX_RADIAL + 1)
    C_real_FG     = (LMAX_FG + 1)**2 + (LMAX_FG + 1)
    C_real_BG     = (LMAX_BG + 1)**2 + (LMAX_BG + 1)

    # --- Define slicing indices based on the REAL vector sizes ---
    start_idx_rad = 0
    end_idx_rad   = start_idx_rad + C_real_RADIAL
    start_idx_fg  = end_idx_rad
    end_idx_fg    = start_idx_fg + C_real_FG * N_PHASE_MAPS * N_FG_CLASSES_PER_MAP
    start_idx_bg  = end_idx_fg
    end_idx_bg    = start_idx_bg + C_real_BG * N_PHASE_MAPS

    def reconstruct_all_maps(state_vec_gpu, stats_dict):
        maps = {}
        state_vec = state_vec_gpu.cpu().numpy().squeeze(0)

        # ---- 1. Reconstruct Radial Map ----
        radial_real_coeffs = torch.from_numpy(state_vec[start_idx_rad:end_idx_rad])
        radial_real_coeffs_3d = radial_real_coeffs.reshape(1, C_pos_RADIAL, 2)
        radial_cplx_coeffs_norm = real_to_complex_channel_stack(radial_real_coeffs_3d)
        radial_coeffs_orig = unnorm_healpix_alms_final(radial_cplx_coeffs_norm, stats_dict["radial"])[0, :, 0]
        maps['radial'] = hp.alm2map(radial_coeffs_orig.astype(np.complex128), nside=NSIDE_RADIAL, lmax=LMAX_RADIAL)

        # ---- 2. Reconstruct Phase Maps ----
        fg_real_flat = state_vec[start_idx_fg:end_idx_fg]
        bg_real_flat = state_vec[start_idx_bg:end_idx_bg]
        fg_real_coeffs = fg_real_flat.reshape(-1, N_PHASE_MAPS * N_FG_CLASSES_PER_MAP)
        bg_real_coeffs = bg_real_flat.reshape(-1, N_PHASE_MAPS)

        fg_maps, bg_maps = [], []
        for i in range(N_PHASE_MAPS):
            bg_real_vec = torch.from_numpy(bg_real_coeffs[:, i])
            bg_real_vec_3d = bg_real_vec.reshape(1, C_pos_BG, 2)
            bg_cplx_norm = real_to_complex_channel_stack(bg_real_vec_3d)
            bg_cplx_orig = unnorm_healpix_alms_final(bg_cplx_norm, stats_dict["phase_bg"])[0, :, 0]
            bg_maps.append(hp.alm2map(bg_cplx_orig.astype(np.complex128), nside=NSIDE_PHASE, lmax=LMAX_BG).real)

            for j in range(N_FG_CLASSES_PER_MAP):
                ch_idx = i * N_FG_CLASSES_PER_MAP + j
                fg_real_vec = torch.from_numpy(fg_real_coeffs[:, ch_idx])
                fg_real_vec_3d = fg_real_vec.reshape(1, C_pos_FG, 2)
                fg_cplx_norm = real_to_complex_channel_stack(fg_real_vec_3d)
                fg_cplx_orig = unnorm_healpix_alms_final(fg_cplx_norm, stats_dict["phase_fg"])[0, :, 0]
                fg_maps.append(hp.alm2map(fg_cplx_orig.astype(np.complex128), nside=NSIDE_PHASE, lmax=LMAX_FG).real)

        for i in range(N_PHASE_MAPS):
            component_list = [bg_maps[i]] + fg_maps[i*N_FG_CLASSES_PER_MAP:(i+1)*N_FG_CLASSES_PER_MAP]
            maps[f'phase_{i+1}'] = np.argmax(np.stack(component_list, axis=0), axis=0).astype(float)
            
        return maps

    # ==============================================================================
    # 6. NEW: Main Evaluation Loop
    # ==============================================================================
    all_results = []

    for i, folder_name in enumerate(val_folds):
        print(f"\nProcessing folder {i+1}/{len(val_folds)}: {folder_name}")
        
        # --- a. Load data for the current folder ---
        # We find the index in the dataset corresponding to the folder name
        try:
            idx = ds_val.folder_list.index(folder_name)
            traj = ds_val[idx]
        except ValueError:
            print(f"  ► WARNING: Folder '{folder_name}' not found in dataset. Skipping.")
            continue
            
        T = traj["radial"].shape[0]
        
        # --- b. Prepare initial state and conditioning ---
        seq = torch.cat([merge_coeffs(traj[k].unsqueeze(0)) 
                        for k in ["radial", "fg", "bg"]], 2).to(device)
        cond = traj["cond"].unsqueeze(0).to(device)
        
        # We start prediction from the first frame
        start_frame = 0
        x_current = seq[:, start_frame]

        # --- c. Auto-regressive generation and evaluation loop ---
        # Determine how many steps to predict for this specific trajectory
        num_predictions = min(MAX_PREDICTION_HORIZON, T - 1)
        
        for step in range(num_predictions):
            current_frame_idx = start_frame + step
            next_frame_idx = current_frame_idx + 1
            
            # This is the conditioning frame for the SDE
            x_t = seq[:, current_frame_idx] 
            
            # --- SDE generation for the next frame (timed) ---
            t0_gen = time.perf_counter()
            X_sde = torch.zeros(SDE_STEPS + 1, 1, D, device=device)
            X_sde[0] = x_t.unsqueeze(0)
            taus = torch.linspace(0, 1.0, SDE_STEPS, device=device).unsqueeze(-1).unsqueeze(-1)
            
            with torch.no_grad():
                for k, tau in enumerate(taus):
                    x_tau = X_sde[k]
                    x_tau3d = x_tau.unsqueeze(1)
                    z_tau = torch.randn_like(x_tau3d)
                    
                    b_hat, eta_hat = net(x_t=x_t.unsqueeze(1), x_tau=x_tau3d, cfg=cond, tau=tau)
                    
                    gamma = gamma_tau(tau)
                    noise_modulator = 1.2 * eps_const / (gamma + 0.05)
                    drift = (b_hat.squeeze(1) - noise_modulator * eta_hat.squeeze(1)) / SDE_STEPS
                    diff  = math.sqrt(2.0) * z_tau.squeeze(1)
                    
                    X_sde[k+1] = x_tau + drift + diff
            predicted_state = X_sde[-1]
            t1_gen = time.perf_counter()
            gen_duration = t1_gen - t0_gen
            
            # --- d. Reconstruct predicted and true maps (timed) ---
            t0_rec = time.perf_counter()
            predicted_maps = reconstruct_all_maps(predicted_state, all_stats)
            true_state = seq[:, next_frame_idx]
            true_maps = reconstruct_all_maps(true_state, all_stats)
            t1_rec = time.perf_counter()
            rec_duration = t1_rec - t0_rec
            
            # --- e. Calculate and store metrics ---
            frame_metrics = calculate_metrics(predicted_maps, true_maps)
            frame_metrics['folder'] = folder_name
            frame_metrics['frame'] = next_frame_idx  # We predicted the state for frame t+1
            
            # Timing metrics
            frame_metrics['gen_time_s'] = gen_duration
            frame_metrics['rec_time_s'] = rec_duration
            frame_metrics['total_time_s'] = gen_duration + rec_duration
            
            all_results.append(frame_metrics)
            
            # --- f. Log progress ---
            mae = frame_metrics['radial_mae']
            print(f"  ► Frame {next_frame_idx:03d} predicted. Radial MAE: {mae:.4f} | total_time_s: {frame_metrics['total_time_s']:.4f}s")

            # For the next iteration, the new "current" state is the ground truth
            x_current = true_state

    print("\n✅ Evaluation finished for all folders.")

    # ==============================================================================
    # 7. NEW: Save Results to CSV
    # ==============================================================================
    if all_results:
        # Get the header from the keys of the last dictionary entry
        header = list(all_results[-1].keys())
        # Reorder to have folder and frame first
        header.sort(key=lambda x: (x != 'folder', x != 'frame'))

        try:
            with open(RESULTS_FILE, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(all_results)
            print(f"✅ Results successfully saved to {RESULTS_FILE}")
        except IOError:
            print(f"❌ Error: Could not write to {RESULTS_FILE}")
    else:
        print("No results were generated to save.")

if __name__ == "__main__":
    main()