# ──────────────────────────────────────────────────────────────────────────────
#  Stochastic Interpolant for Lipid Diffusion
# ──────────────────────────────────────────────────────────────────────────────


rotations = 0
model_name = f"big_{rotations}rots"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast   # ← only this
from typing import List, Tuple
import healpy as hp
import os

# Assuming these are your custom local files
from dataset import HealpixSequenceDataset, collate_complex
from utils import CONFIG_KEYS
from rotated_dataset import RotatedHealpixDataset

import logging, datetime, sys
ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_name = f"train_{model_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_name),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# --- CORRECTED & CENTRALIZED TOKEN WIDTH DEFINITION ---
# This should be the single source of truth for your token dimensions.
RAD_LMAX, FG_LMAX, BG_LMAX = 14, 22, 44
N_RAD, N_FG, N_BG = 2, 18, 6  # Correct map counts for REAL data

# Calculate widths including the *2 factor for real/imaginary parts
RAD_WIDTHS = [(l + 1) * N_RAD for l in range(RAD_LMAX + 1)]
FG_WIDTHS  = [(l + 1) * N_FG  for l in range(FG_LMAX + 1)]
BG_WIDTHS  = [(l + 1) * N_BG  for l in range(BG_LMAX + 1)]

# The final, authoritative list of all 83 token widths
TOKEN_WIDTHS: List[int] = RAD_WIDTHS + FG_WIDTHS + BG_WIDTHS
#print(RAD_WIDTHS)

# ── γ(τ) and derivative ───────────────────────────────────────────────────────
def gamma_tau(t):  # t: [B,1] real
    return (t * (1. - t))

def gamma_dot(t):
    return (1.0 - 2.0 * t)


class TimestepEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding module, from "Attention Is All You Need".
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        if self.dim % 2 != 0:
            raise ValueError("Cannot use sinusoidal embeddings with an odd dimension.")

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): A 1D tensor of timesteps, shape [B].
        Returns:
            torch.Tensor: The time embeddings, shape [B, dim].
        """
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class ConditioningMLP(nn.Module):
    """
    An MLP to process and fuse the tau and configuration conditioning signals.
    """
    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        # Input to this MLP will be the concatenation of the two embeddings
        input_dim = embed_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(), # Swish/SiLU is a common choice here
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, tau_embed: torch.Tensor, cond_embed: torch.Tensor) -> torch.Tensor:
        combined_embed = torch.cat([tau_embed, cond_embed], dim=1)
        return self.mlp(combined_embed)



# -----------------------------------------------------------------------------
#  Diffusion Transformer with tokenised input (unchanged DiTBlock API).
# -----------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Exactly the same as the user’s original DiTBlock."""
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, n_layers: int = 6):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.h = hidden_size
        self.num_layers = n_layers
        
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1), # <-- ADD DROPOUT HERE
            nn.Linear(mlp_hidden, hidden_size),
            #nn.Dropout(0.1)  # <-- AND/OR HERE
        )
        self.mod   = nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        self._init_adaln_zero()
    
    # -------------------------------------------------------------
    #  AdaLN-Zero initialisation  (DiT++ leak α = 1/√L)
    # -------------------------------------------------------------
    def _init_adaln_zero(self):
        with torch.no_grad():                    # <-- key line
            #   weight & bias are [6H × H] and [6H] tensors
            self.mod.weight.zero_()
            self.mod.bias.zero_()

            # γ₁, γ₂  = 1
            self.mod.bias[0*self.h : 1*self.h].fill_(1.0)
            self.mod.bias[2*self.h : 3*self.h].fill_(1.0)

            # α₁, α₂  = 1/√L
            #k = 0.0 #/ math.sqrt(self.num_layers)
            #self.mod.bias[4*self.h : 5*self.h].fill_(k)
            #self.mod.bias[5*self.h : 6*self.h].fill_(k)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mod_out = self.mod(F.silu(cond))
        g1, b1, g2, b2, a1, a2 = mod_out.chunk(6, dim=1)

        """
        print(x.shape)

        # In your DiTBlock's forward method, after `print(h.shape)`

        # It's good practice to do debugging prints inside a no_grad() context
        with torch.no_grad():
            # Let's inspect tokens for the first item in the batch
            b_idx = 0 
            
            # --- Inspect Token 0 ---
            token_0 = x[b_idx, 0, :]
            print("\n--- Sanity Check input X: Token 0 ---")
            print(f"  Shape: {token_0.shape}")
            print(f"  Mean:  {token_0.mean().item():.6f}")
            print(f"  Std:   {token_0.std().item():.6f}")
            print(f"  Min:   {token_0.min().item():.6f}")
            print(f"  Max:   {token_0.max().item():.6f}")
            # Print the first 8 values to get a feel for the content
            print(f"  First 8 values: {token_0[:8]}")

            # --- Inspect Token 41 ---
            token_41 = x[b_idx, 41, :]
            print("\n--- Sanity Check input X: Token 41 ---")
            print(f"  Shape: {token_41.shape}")
            print(f"  Mean:  {token_41.mean().item():.6f}")
            print(f"  Std:   {token_41.std().item():.6f}")
            print(f"  Min:   {token_41.min().item():.6f}")
            print(f"  Max:   {token_41.max().item():.6f}")
            # Print the first 8 values
            print(f"  First 8 values: {token_41[:8]}")
            print("-" * 30 + "\n")

            # This will prevent the program from continuing, useful for quick debugging
            # import sys; sys.exit()
        """


        h = g1.unsqueeze(1) * self.norm1(x) + b1.unsqueeze(1)

        """
        print(h.shape)

        # In your DiTBlock's forward method, after `print(h.shape)`

        # It's good practice to do debugging prints inside a no_grad() context
        with torch.no_grad():
            # Let's inspect tokens for the first item in the batch
            b_idx = 0 
            
            # --- Inspect Token 0 ---
            token_0 = h[b_idx, 0, :]
            print("\n--- Sanity Check Pre-Transformer: Token 0 ---")
            print(f"  Shape: {token_0.shape}")
            print(f"  Mean:  {token_0.mean().item():.6f}")
            print(f"  Std:   {token_0.std().item():.6f}")
            print(f"  Min:   {token_0.min().item():.6f}")
            print(f"  Max:   {token_0.max().item():.6f}")
            # Print the first 8 values to get a feel for the content
            print(f"  First 8 values: {token_0[:8]}")

            # --- Inspect Token 41 ---
            token_41 = h[b_idx, 41, :]
            print("\n--- Sanity Check Pre-Transformer: Token 41 ---")
            print(f"  Shape: {token_41.shape}")
            print(f"  Mean:  {token_41.mean().item():.6f}")
            print(f"  Std:   {token_41.std().item():.6f}")
            print(f"  Min:   {token_41.min().item():.6f}")
            print(f"  Max:   {token_41.max().item():.6f}")
            # Print the first 8 values
            print(f"  First 8 values: {token_41[:8]}")
            print("-" * 30 + "\n")

            # This will prevent the program from continuing, useful for quick debugging
            # import sys; sys.exit()
        
        """
        h, _ = self.attn(h, h, h)

        """
        print(h.shape)

        # In your DiTBlock's forward method, after `print(h.shape)`

        # It's good practice to do debugging prints inside a no_grad() context
        with torch.no_grad():
            # Let's inspect tokens for the first item in the batch
            b_idx = 0 
            
            # --- Inspect Token 0 ---
            token_0 = h[b_idx, 0, :]
            print("\n--- Sanity Check Post Transformer: Token 0 ---")
            print(f"  Shape: {token_0.shape}")
            print(f"  Mean:  {token_0.mean().item():.6f}")
            print(f"  Std:   {token_0.std().item():.6f}")
            print(f"  Min:   {token_0.min().item():.6f}")
            print(f"  Max:   {token_0.max().item():.6f}")
            # Print the first 8 values to get a feel for the content
            print(f"  First 8 values: {token_0[:8]}")

            # --- Inspect Token 41 ---
            token_41 = h[b_idx, 41, :]
            print("\n--- Sanity Check Post Transformer: Token 41 ---")
            print(f"  Shape: {token_41.shape}")
            print(f"  Mean:  {token_41.mean().item():.6f}")
            print(f"  Std:   {token_41.std().item():.6f}")
            print(f"  Min:   {token_41.min().item():.6f}")
            print(f"  Max:   {token_41.max().item():.6f}")
            # Print the first 8 values
            print(f"  First 8 values: {token_41[:8]}")
            print("-" * 30 + "\n")

        """
        h = self.attn_drop(h)
        x = x + a1.unsqueeze(1) * h

        h = g2.unsqueeze(1) * self.norm2(x) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.ffn(h)
        return x



class EquiJumpDiT(nn.Module):
    """
    Tokenised version with pre-computed indices for maximum efficiency.
    """
    def __init__(
        self,
        D: int,
        dims: Tuple[int, int, int],
        cond_dim: int = 12,
        cond_dim_emb: int = 256,
        token_size: int = 384,
        n_layers: int = 12,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.rad_dim, self.fg_dim, self.bg_dim = dims
        self.token_size = token_size

        # --- CHANGED: Pre-compute index lookup tables ONCE during initialization ---
        # 1. Get the l_values from healpy (on CPU)
        l_vals_rad, _ = hp.Alm.getlm(lmax=RAD_LMAX)
        l_vals_fg, _ = hp.Alm.getlm(lmax=FG_LMAX)
        l_vals_bg, _ = hp.Alm.getlm(lmax=BG_LMAX)

        # 2. Define a helper to build the non-trainable index lists
        def build_idx_list(l_vals, l_max):
            # Using nn.ParameterList with requires_grad=False is a clean way to
            # store a list of tensors that are part of the model's state
            # and should be moved to the correct device with .to(device).
            return nn.ParameterList([
                nn.Parameter(
                    torch.from_numpy((l_vals == L).nonzero()[0]).long(),
                    requires_grad=False
                ) for L in range(l_max + 1)
            ])

        # 3. Create and store the index lists as part of the model
        self.idx_rad = build_idx_list(l_vals_rad, RAD_LMAX)
        self.idx_fg  = build_idx_list(l_vals_fg, FG_LMAX)
        self.idx_bg  = build_idx_list(l_vals_bg, BG_LMAX)
        # The old `register_buffer` calls for l_vals_* are no longer needed.
        # --- END OF CHANGES ---

        # ------------------------------------------------------------------
        # Per‑token projectors (Unchanged)
        # ------------------------------------------------------------------
        self.token_embedders = nn.ModuleList([
            nn.Linear(2 * w, token_size) for w in TOKEN_WIDTHS
        ])
        self.token_decoders = nn.ModuleList(
            nn.Linear(token_size, 2 *w) for w in TOKEN_WIDTHS
        )

        # ------------------------------------------------------------------
        # Conditioning MLP (Unchanged)
        # ------------------------------------------------------------------
        embed_dim = 128
        self.tau_embed   = TimestepEmbedding(embed_dim)
        self.cond_embed  = nn.Linear(cond_dim, embed_dim)
        self.cond_fuse   = ConditioningMLP(embed_dim, cond_dim_emb)

        # ------------------------------------------------------------------
        # Core Transformer (Unchanged)
        # ------------------------------------------------------------------
        self.blocks = nn.ModuleList([DiTBlock(hidden_size=token_size, num_heads=n_heads, cond_dim=cond_dim_emb) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(token_size, elementwise_affine=False, eps=1e-6)

    def tokenize(self, x_t, x_tau):
        x_t_flat   = x_t.squeeze(1)
        x_tau_flat = x_tau.squeeze(1)

        radial_flat_xt, fg_flat_xt, bg_flat_xt = torch.split(x_t_flat, (self.rad_dim, self.fg_dim, self.bg_dim), dim=-1)
        radial_flat_xtau, fg_flat_xtau, bg_flat_xtau = torch.split(x_tau_flat, (self.rad_dim, self.fg_dim, self.bg_dim), dim=-1)

        def add_channels(x, n_chan: int):
            d = x.shape[-1]
            return x.reshape(*x.shape[:-1], d // n_chan, n_chan)

        radial_xt  = add_channels(radial_flat_xt,  N_RAD)
        radial_tau = add_channels(radial_flat_xtau, N_RAD)
        fg_xt      = add_channels(fg_flat_xt,  N_FG)
        fg_tau     = add_channels(fg_flat_xtau, N_FG)
        bg_xt      = add_channels(bg_flat_xt,  N_BG)
        bg_tau     = add_channels(bg_flat_xtau, N_BG)

        def tokenize_one(rad_x, rad_tau, fg_x, fg_tau, bg_x, bg_tau):
            # --- CHANGED: Use pre-computed indices for direct lookup ---
            # Instead of searching for indices with `coeffs_for_degreegpu`, we
            # directly use the stored lists for a much faster `index_select`.
            radial_tokens_x   = [rad_x.index_select(1, self.idx_rad[L]) for L in range(RAD_LMAX + 1)]
            radial_tokens_tau = [rad_tau.index_select(1, self.idx_rad[L]) for L in range(RAD_LMAX + 1)]

            fg_tokens_x       = [fg_x.index_select(1, self.idx_fg[L]) for L in range(FG_LMAX + 1)]
            fg_tokens_tau     = [fg_tau.index_select(1, self.idx_fg[L]) for L in range(FG_LMAX + 1)]

            bg_tokens_x       = [bg_x.index_select(1, self.idx_bg[L]) for L in range(BG_LMAX + 1)]
            bg_tokens_tau     = [bg_tau.index_select(1, self.idx_bg[L]) for L in range(BG_LMAX + 1)]
            # --- END OF CHANGES ---

            all_x_tokens = radial_tokens_x + fg_tokens_x + bg_tokens_x
            all_tau_tokens = radial_tokens_tau + fg_tokens_tau + bg_tokens_tau
            
            projected_tokens = []
            for i in range(len(all_x_tokens)):
                tok_x = all_x_tokens[i]
                tok_tau = all_tau_tokens[i]
                
                tok_combined = torch.cat([tok_x, tok_tau], dim=2)
                
                B, L_plus_1, C_doubled = tok_combined.shape
                tok_flat = tok_combined.reshape(B, -1)
                
                projected = self.token_embedders[i](tok_flat)
                projected_tokens.append(projected)
                
            h = torch.stack(projected_tokens, dim=1)
            return h

        h = tokenize_one(
            radial_xt, radial_tau,
            fg_xt, fg_tau,
            bg_xt, bg_tau
        )
        return h

    def forward(self, x_t: torch.Tensor, x_tau: torch.Tensor, cfg: torch.Tensor, tau: torch.Tensor):
        # This part remains unchanged
        h = self.tokenize(x_t, x_tau)
        c_vec = self.cond_fuse(self.tau_embed(tau.squeeze(-1)), self.cond_embed(cfg))
        for blk in self.blocks:
            h = checkpoint(blk, h, c_vec, use_reentrant=False)
        h = self.final_ln(h)
        b_hat_tokens, eta_hat_tokens = [], []
        for i in range(h.size(1)):
            projected = self.token_decoders[i](h[:, i])
            b_tok, eta_tok = projected.chunk(2, dim=-1)
            b_hat_tokens.append(b_tok)
            eta_hat_tokens.append(eta_tok)
        b_hat = torch.cat(b_hat_tokens, dim=1)
        eta_hat = torch.cat(eta_hat_tokens, dim=1)
        return b_hat, eta_hat


class PrevDifference(nn.Module):
    """Returns all-zero b̂ and η̂ with the correct shapes."""
    def forward(self, x_t, x_tau, cfg, tau):
        return (x_tau - x_t), torch.zeros_like(x_t.squeeze(1))
    
class PrevFrame(nn.Module):
    """Returns all-zero b̂ and η̂ with the correct shapes."""
    def forward(self, x_t, x_tau, cfg, tau):
        return (x_t), torch.zeros_like(x_t.squeeze(1))


def inner(a, b):
    """batch-wise dot product along last axis"""
    return (a * b).sum(dim=-1)            # shape (B,)

# ── EquiJump loss (with configurable regularization weight) ──────────────────
def equijump_loss(target_b, target_eta, b_hat, eta_hat, reg_weight=0.05):
    #print(target_b.shape, target_eta.shape, b_hat.shape, eta_hat.shape)
    """
    Implements Eq.(9) of EquiJump for *real* tensors.

      L = λ/2 (‖b̂‖² + ‖η̂‖²) − b_target·b̂ − η_target·η̂
    """
    term_b   = 0.125 * inner(b_hat,  b_hat)  - inner(target_b,  b_hat)
    term_eta = 0.125 * inner(eta_hat, eta_hat) - inner(target_eta, eta_hat)

    #random stuff to debug
    # Epsilon for numerical stability to prevent division by zero for zero-vectors.
    epsilon = 1e-8

    # --- Target Calculation ---
    target_b_norm = torch.norm(target_b, p=2, dim=-1, keepdim=True)
    target_eta_norm = torch.norm(target_eta, p=2, dim=-1, keepdim=True)
    
    target_b_dir = target_b / (target_b_norm + epsilon)
    target_eta_dir = target_eta / (target_eta_norm + epsilon)

    # --- Prediction Calculation ---
    pred_b_norm = torch.norm(b_hat, p=2, dim=-1, keepdim=True)
    pred_eta_norm = torch.norm(eta_hat, p=2, dim=-1, keepdim=True)

    pred_b_dir = b_hat / (pred_b_norm + epsilon)
    pred_eta_dir = eta_hat / (pred_eta_norm + epsilon)

    dir_loss_b = (1 - F.cosine_similarity(pred_b_dir, target_b_dir, dim=-1)).mean()
    dir_loss_eta = (1 - F.cosine_similarity(pred_eta_dir, target_eta_dir, dim=-1)).mean()
    dot_beta = inner(b_hat, target_b)
    dot_eta = inner(eta_hat, target_eta)

    # Magnitude Loss: Use L1 loss for robustness.
    # The norms are already calculated per-batch element, so we just take the mean.
    mag_loss_b = reg_weight * inner(b_hat,  b_hat)
    mag_loss_eta = 5 * reg_weight * inner(eta_hat, eta_hat)
    magnitude_loss = (mag_loss_b + mag_loss_eta) * 0.1  # Scale by magnitude weight

    loss = (
        term_b.mean() + term_eta.mean()      # dot-product part
      #+ 5.0   * dir_loss_b.mean()   # NEW: b-direction
      #+ 1.0 * dir_loss_eta.mean() # keep 0 for now
    )


    return loss, dir_loss_b, dir_loss_eta, mag_loss_b, mag_loss_eta, pred_b_norm, target_b_norm, pred_eta_norm, target_eta_norm, dot_beta, dot_eta     # scalar
    

# ── "Reality Check" Plotting Function ────────────────────────────────────────
def reality_check_plot(b_hat, target_b, epoch, filename="reality_check.png"):
    # Detaching and moving to CPU for numpy operations
    b_hat_real = b_hat.real.detach().cpu().numpy().flatten()
    target_b_real = target_b.real.detach().cpu().numpy().flatten()
    b_hat_imag = b_hat.imag.detach().cpu().numpy().flatten()
    target_b_imag = target_b.imag.detach().cpu().numpy().flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Distribution Reality Check - Epoch {epoch}', fontsize=16)

    ax1.hist(target_b_real, bins=50, density=True, color='blue', alpha=0.6, label='Target (Reality)')
    ax1.hist(b_hat_real, bins=50, density=True, color='red', alpha=0.6, label='Prediction (b_hat)')
    ax1.set_title('Real Part')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(target_b_imag, bins=50, density=True, color='blue', alpha=0.6, label='Target (Reality)')
    ax2.hist(b_hat_imag, bins=50, density=True, color='red', alpha=0.6, label='Prediction (b_hat)')
    ax2.set_title('Imaginary Part')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)


# --- REWRITTEN step_batch function for Multi-Step Objective ---
def step_batch(model, batch, device, epoch, mag_weight, noise_level=0.01, unroll_steps=1):
    def flatten_block(x):
        B, T, C, M = x.shape
        return x.reshape(B, T, C * M)

    radial = flatten_block(batch["radial"]).to(device)  # (B, T, 240)
    fg     = flatten_block(batch["fg"]).to(device)      # (B, T, 4968)
    bg     = flatten_block(batch["bg"]).to(device)      # (B, T, 6210)

    # Merge per timestep
    seq = torch.cat([radial, fg, bg], dim=-1)  # (B, T, D)
    cond = batch["cond"].to(device)
    B, T, D = seq.shape
    
    # We need enough frames for the unrolling
    if T <= unroll_steps:
        return torch.tensor(0.0, device=device), None, None, None, None # Skip batch if too short

    # Create pairs of all possible start and end points for the unroll
    x_t_initial = seq[:, :-(unroll_steps), :]
    
    B_eff = B * (T - 1 - (unroll_steps-1) )
    x_t_initial = x_t_initial.reshape(B_eff, 1, D)
    cond = cond.repeat_interleave(T - 1 - (unroll_steps-1), dim=0)
    
    # --- Multi-step loss accumulation ---
    total_loss = 0.0
    current_x_pred = x_t_initial # Start with the ground truth
    
    # Keep track of the last predictions for logging
    last_b_hat, last_eta_hat, last_target_b, last_target_eta = (None,) * 4

    for k in range(unroll_steps):
        x_t_k = current_x_pred # Use previous prediction (or initial state) as input
        x_tp1_k_true = seq[:, k+1:k+1+(T-1-(unroll_steps-1)), :].reshape(B_eff, 1, D)
        #print(f"Step {k+1}/{unroll_steps}: x_t_k shape = {x_t_k.shape}, x_tp1_k_true shape = {x_tp1_k_true.shape}")
        
        # --- Noise Injection for Robustness ---
        if noise_level > 0 and k == 0: # Only add noise to the very first step
            x_t_k = x_t_k + torch.randn_like(x_t_k) * noise_level

        tau = torch.rand(B_eff, 1, device=device)
        
        delta   = (x_t_k - x_tp1_k_true)                      # shape B×1×D
        sigma   = 0.10 * delta.norm(dim=-1, keepdim=True) / math.sqrt(delta.shape[-1])
        #print(sigma)
                                         # B×1×1  – per-sequence scale
 
        #z_tau = torch.randn_like(x_t)
        z_tau = torch.randn_like(x_t_k) * sigma      # small Gaussian bump

        #z_tau = torch.randn_like(x_t_k)
        x_tau_k = (1-tau.unsqueeze(-1))*x_t_k + tau.unsqueeze(-1)*x_tp1_k_true + gamma_tau(tau).unsqueeze(-1)*z_tau

        #print(x_t_k.shape, x_tau_k.shape, tau.shape, cond.shape, D)
    
        b_hat, eta_hat = model(x_t_k, x_tau_k, cond, tau)
        
        target_b = (x_tp1_k_true - x_t_k) + gamma_dot(tau).unsqueeze(-1) * z_tau
        target_eta = z_tau
        
        # Accumulate loss from this step
        loss, dir_loss_b, dir_loss_eta, mag_loss_b, mag_loss_eta, pred_b_norm, target_b_norm, pred_eta_norm, target_eta_norm, dot_beta, dot_eta  = equijump_loss(target_b.squeeze(1), target_eta.squeeze(1), b_hat, eta_hat, reg_weight=mag_weight)

        total_loss = total_loss + loss
        #log.info(f"loss = {loss.item():.4f}, dir_loss = {dir_loss.item():.4f}, mag_loss = {mag_loss.item():.4f}")

        # Update the predicted state for the next unroll step
        # We detach it so the gradient from step k+1 doesn't flow into the prediction at step k
        # This is a common simplification known as "teacher forcing"-like unrolling.
        current_x_pred = (x_t_k + b_hat - gamma_dot(tau).unsqueeze(-1) * z_tau).detach()
        
        # Save the tensors from the last step for the reality check plot
        if k == unroll_steps - 1:
            last_b_hat, last_eta_hat = b_hat, eta_hat
            last_target_b, last_target_eta = target_b, target_eta
            
    # Average the loss over the number of unrolled steps
    final_loss = total_loss / unroll_steps

    #return (final_loss, last_b_hat.squeeze(1), last_eta_hat.squeeze(1), 
    #        last_target_b.squeeze(1), last_target_eta.squeeze(1))
    return (loss, dir_loss_b, dir_loss_eta, mag_loss_b, mag_loss_eta, pred_b_norm, target_b_norm, pred_eta_norm, target_eta_norm, dot_beta, dot_eta,
            last_b_hat.squeeze(1), last_eta_hat.squeeze(1), 
            last_target_b.squeeze(1), last_target_eta.squeeze(1))


@torch.no_grad()
def eval_loss(model, loader, device, reg_weight=0.1):
    """Average EquiJump loss over a DataLoader (no grad)."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        # ---- copy the *target* construction from step_batch, but only once ----
        def flat(x):  # (B,T,C,M) -> (B,T,C*M)
            B,T,C,M = x.shape; return x.reshape(B, T, C*M)
        radial = flat(batch["radial"]).to(device)
        fg     = flat(batch["fg"]).to(device)
        bg     = flat(batch["bg"]).to(device)
        seq  = torch.cat([radial, fg, bg], dim=-1)          # (B,T,D)
        cond = batch["cond"].to(device)
        B,T,D = seq.shape
        if T < 2: continue                                  # skip too-short seqs

        x_t   = seq[:, 0:1]                                 # (B,1,D)
        x_t1  = seq[:, 1:2]
        tau   = torch.rand(B, 1, device=device)
        delta = (x_t - x_t1)
        sigma = 0.10 * delta.norm(dim=-1, keepdim=True) / math.sqrt(D)
        z_tau = torch.randn_like(x_t) * sigma
        x_tau = (1 - tau)*x_t + tau*x_t1 + gamma_tau(tau)*z_tau

        # ---------------- baseline prediction -----------------
        b_hat, eta_hat = model(x_t, x_tau, cond, tau)       # all zeros here

        target_b   = (x_t1 - x_t) + gamma_dot(tau)*z_tau
        target_eta = z_tau

        loss = equijump_loss(
            target_b.squeeze(1), target_eta.squeeze(1),
            b_hat, eta_hat, reg_weight=reg_weight
        )[0]                                               # first item is scalar
        total += loss.item() * B
        n     += B
    return total / n



# ── the training harness ──────────────────────────────────────────────────────
def train_once(root=".",
               stats="sh_norm_stats_correct.pkl",
               train_list="training_folders_final.txt",
               val_list='val_folders_final.txt',
               epochs=50,
               rotations = rotations,
               batch_size=3,
               lr=1e-5,
               reg_weight=0.5,
               noise_level=0.01, # New hyperparameter for noise injection
               unroll_prob = 0.0,
               unroll_steps = 1,
               device="cuda"):


    with open(train_list) as f:
        train_folders = [ln.strip() for ln in f if ln.strip()]
    with open(val_list) as f:
        val_folders = [ln.strip() for ln in f if ln.strip()]

    ### START: ADD THIS BLOCK ###
    log_file = f"training_log_{model_name}.csv"
    # Write header if the file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            ### FIX: Add the new column names for the norms ###
            f.write("epoch,train_loss,val_loss,pred_b_norm,target_b_norm,pred_eta_norm,target_eta_norm\n")


    train = HealpixSequenceDataset(root_path=root, folder_list=train_folders, stats_path=stats)
    val = HealpixSequenceDataset(root_path=root, folder_list=val_folders, stats_path=stats)
    train_ds = RotatedHealpixDataset(train, k_rot=rotations)
    val_ds = val
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_complex)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_complex)
    #print('dataset sizes:', len(train_ds), len(val_ds))
    sample = train_ds[0]
    
    n_samples = len(train_ds)           # total training examples (after augmentation)
    n_batches = len(train_dl)           # optimiser steps per epoch

    log.info(
        f"Model: {model_name} | "
        f"Samples: {n_samples:,} | "
        f"Batches/epoch: {n_batches:,} | "
        f"Rotations: {rotations}"
    )

    #print(sample["radial"].shape, sample["fg"].shape, sample["bg"].shape)
    nC = lambda arr: arr.shape[1] * arr.shape[2]   # include ×2
    D = nC(sample["radial"]) + nC(sample["fg"]) + nC(sample["bg"])
    P = len(CONFIG_KEYS)
    #print(f"Total dimension D = {D} (radial + fg + bg) + {P} (conditioning) = {D + P}")
    rad_dim = nC(sample["radial"])
    fg_dim  = nC(sample["fg"])
    bg_dim  = nC(sample["bg"])
    dims = (rad_dim, fg_dim, bg_dim)



    #baseline 
    device = "cuda"
    zero_model = PrevDifference().to(device)
    train_zero = eval_loss(zero_model, train_dl, device)
    val_zero   = eval_loss(zero_model, val_dl,   device)
    log.info(f"[baseline] prevdifference-predictor   train={train_zero:.3f}   val={val_zero:.3f}")

    device = "cuda"
    zero_model = PrevFrame().to(device)
    train_zero = eval_loss(zero_model, train_dl, device)
    val_zero   = eval_loss(zero_model, val_dl,   device)
    log.info(f"[baseline] prevframe-predictor   train={train_zero:.3f}   val={val_zero:.3f}")
    # ------------------------------------------------------------------



    model = EquiJumpDiT(D, cond_dim=P, dims=dims).to(device)
    with torch.no_grad():
        for m in model.token_embedders: m.weight.mul_(1e-1)
        for m in model.token_decoders:  m.weight.mul_(1e-1)

    #model = torch.compile(
    #    model,
    #    mode="reduce-overhead",   # good default for training loops
    #    fullgraph=False)          # keep small python-side ops outside the graph
    
    # ------------------------------------------------------------------
    #  Optimiser FIRST  ★
    # ------------------------------------------------------------------
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # ------------------------------------------------------------------
    #  Scheduler: 1 k warm-up  →  cosine decay  ★
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_dl)
    total_steps     = epochs * steps_per_epoch

    warmup_steps   = 1_000
    min_lr         = 2.5e-6
    main_steps     = max(total_steps - warmup_steps, 1)

    sched_warm = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    # -- 2. Cosine + restarts ------------------------------------------------------
    first_cycle   = 5 * steps_per_epoch   # 5 epochs
    sched_main = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim,
        T_0     = first_cycle,   # length of the 1st cycle
        T_mult  = 2,             # cycle length doubles each time
        eta_min = 2.5e-6
    )

    sched = torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[sched_warm, sched_main], milestones=[warmup_steps]
    )

    # ------------------------------------------------------------------
    #  Everything else unchanged
    # ------------------------------------------------------------------
    scaler   = torch.amp.GradScaler(device="cuda")
    best_val = float("inf")
    log.info("✅ Stochastic interpolant model initialized.")


    global_step = 0
    for ep in range(epochs):
        train_ds.set_epoch(ep)
        model.train()
        running_loss = 0.0
        
        #total_pairs_processed = 0
        first_batch = True  
        for batch_idx, batch in enumerate(train_dl):
          
            
            # decide stochastically whether to use the long rollout
            if torch.rand(1).item() < unroll_prob:
                u_steps = unroll_steps            # long
            else:
                u_steps = 1                       # ordinary single-step
            
            #print('batch shapes', batch["radial"].shape, batch["fg"].shape, batch["bg"].shape)
            optim.zero_grad()

            with autocast('cuda', enabled=False):
                # Pass this scheduled weight to the step function
                loss, dir_loss_b, dir_loss_eta, mag_loss_b, mag_loss_eta, pred_b_norm, target_b_norm, pred_eta_norm, target_eta_norm, dot_beta, dot_eta, b_hat, eta_hat, target_b, target_eta = step_batch(
                    model, batch, device, epoch=ep,
                    mag_weight=reg_weight, # Use the scheduled weight
                    noise_level=noise_level,
                    unroll_steps = u_steps
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optim)                              # grads are now in FP32

            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 750.0)


            # ------------------------------------------------------------------
            # Turn any vector-valued tensors into scalar summaries (mean-of-batch)
            # ------------------------------------------------------------------
            loss_val        = loss.item()
            dir_b_val       = dir_loss_b.item()
            dir_eta_val     = dir_loss_eta.item()
            mag_b_val       = mag_loss_b.mean().item()
            mag_eta_val     = mag_loss_eta.mean().item()
            grad_val        = grad_norm          # already a Python float
            dot_b_val       = dot_beta.mean().item()
            dot_eta_val     = dot_eta.mean().item()
            bh_norm         = b_hat      .norm().item()         # ‖b̂‖₂  (predicted)
            tgt_bh_norm     = target_b   .norm().item()         # ‖b‖₂   (ground-truth)
            eta_norm        = eta_hat    .norm().item()         # ‖η̂‖₂
            tgt_eta_norm    = target_eta .norm().item()         # ‖η‖₂
           

            # ---------------------------------------------------------------
            # Single, safe log statement (everything is now a true scalar)
            # ---------------------------------------------------------------
            if batch_idx % 10 == 0:
                log.info(
                    f"Batch {batch_idx:06d} | "
                    f"grad norm {grad_val:8.1f} | "
                    f"loss {loss_val:8.3f} | "
                    f"dir_b {dir_b_val:8.3f} | "
                    f"dir_eta {dir_eta_val:8.3f} | "
                    f"mag_b {mag_b_val:8.3f} | "
                    f"mag_eta {mag_eta_val:8.3f} | "
                    f"grad_norm {grad_val:8.1f} | "
                    f"dot_b {dot_b_val:8.3f} | "
                    f"dot_eta {dot_eta_val:8.3f} | "
                    f"‖b̂‖ {bh_norm:8.1f} / {tgt_bh_norm:8.1f} tgt | "
                    f"‖η̂‖ {eta_norm:8.1f} / {tgt_eta_norm:8.1f} tgt"
                )

            scaler.step(optim)
            scaler.update()

            if first_batch:                    # ➋ skip only once
                first_batch = False
            else:
                sched.step()                   #   ordinary behaviour


            global_step += 1
           

            running_loss += loss.item()
            #total_pairs_processed += b_hat.shape[0]
            last_b_norm = torch.norm(b_hat,  p=2).item()
            last_eta_norm = torch.norm(eta_hat, p=2).item()
            last_target_b_norm = torch.norm(target_b, p=2).item()
            last_target_eta_norm = torch.norm(target_eta, p=2).item()
            #torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())


        train_loss = running_loss / len(train_dl)

        log.info("-" * 70)
        log.info(f"End of Epoch {ep:02d}")
        #print(f"  Frames processed this epoch: {total_pairs_processed}")
        
        with torch.no_grad():
           
            
            log.info(f"  Training Loss: {train_loss:.4f}")
            log.info(f"  L2 Norms (b̂) -> Pred: {last_b_norm:.4f} vs Target: {last_target_b_norm:.4f}")
            log.info(f"  L2 Norms (η̂) -> Pred: {last_eta_norm:.4f} vs Target: {last_target_eta_norm:.4f}")


        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                loss , *_ = step_batch(
                    model, batch, device, epoch=ep,
                    noise_level=0.0,
                    mag_weight=reg_weight,
                    unroll_steps = u_steps)
                val_running_loss += loss.item()
        val_loss = val_running_loss / len(val_dl)
        log.info(f"  Validation Loss: {val_loss:.4f}")
        
        # Append the results for the current epoch to the log file
        with open(log_file, 'a') as f:
            ### FIX: Add the four norm variables to the CSV line ###
            f.write(f"{ep},{train_loss},{val_loss},{last_b_norm},{last_target_b_norm},{last_eta_norm},{last_target_eta_norm}\n")


        log.info("-" * 70)

        torch.save(model.state_dict(), f"lipidgen_last_epoch_{model_name}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": ep, "val_loss": val_loss,
                "model_state": model.state_dict(), "opt_state": optim.state_dict(),
            }, f"lipidgen_best_{model_name}.pt")





if __name__ == "__main__":
    train_once()

