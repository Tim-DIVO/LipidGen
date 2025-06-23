#utils.py
import os
import pickle
import numpy as np
import torch
import math
import json
import healpy as hp

#CONFIG STUFF

CONFIG_KEYS = [               # <<– FIXED order everywhere
    "k_bend_saturated",
    "k_bend_unsaturated",
    "k_bend_cholesterol",
    "w_c_default",
    "w_c_U_S",
    "w_c_U_C",
    "w_c_C_S",
    "Temperature",
    "outer_typeOne",
    "outer_typeTwo",
    "inner_typeThr",
    "inner_typeFour",
]


def parse_config_file(config_txt_path: str) -> dict:
    """
    Given the full path to "configuration.txt", return a dict where
    - keys are the left‐hand names (e.g. 'k_bend_saturated'),
    - values are floats, except for 'folder' which remains a string.
    """
    cfg = {}
    with open(config_txt_path, 'r') as f:
        lines = f.readlines()
    # Skip header (first line: "Simulation Configuration:")
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "folder":
            cfg[key] = val
        else:
            try:
                cfg[key] = float(val)
            except ValueError:
                # In case some other non‐float appears, just store as string
                cfg[key] = val
    return cfg


def cfg2vec(cfg_dict, μσ=None):
    """dict ➜ float32 tensor [P].   If μσ given → z-score."""
    vec = np.array([cfg_dict[k] for k in CONFIG_KEYS], dtype=np.float32)
    if μσ is not None:
        μ, σ = μσ                   # each shape (P,)
        vec  = (vec - μ) / σ
    return torch.from_numpy(vec)    # (P,)


def complex_to_real_channel_stack(arr_cplx: torch.Tensor) -> torch.Tensor:
    """
    Converts a complex tensor to a real tensor by stacking the real and
    imaginary parts as new channels along the M dimension.

    This is the recommended method as it perfectly preserves the C dimension and
    its healpy ordering.

    Args:
        arr_cplx (torch.Tensor): Input complex tensor of shape (T, C, M).

    Returns:
        torch.Tensor: Output real tensor of shape (T, C, 2*M).
    """
    if not torch.is_complex(arr_cplx):
        raise TypeError("Input tensor must be complex.")
    if arr_cplx.dim() != 3:
        raise ValueError(f"Input tensor must be 3D (T, C, M), but has "
                         f"{arr_cplx.dim()} dimensions.")
    
    # Separate real and imaginary parts. Both will have shape (T, C, M).
    real_part = arr_cplx.real
    imag_part = arr_cplx.imag
    
    # Concatenate along the LAST dimension (dim=-1 or dim=2) to create 2*M channels.
    return torch.cat([real_part, imag_part], dim=-1)


def real_to_complex_channel_stack(arr_real: torch.Tensor) -> torch.Tensor:
    """
    Converts a real tensor (created by complex_to_real_channel_stack) back to
    a complex tensor.

    Args:
        arr_real (torch.Tensor): Input real tensor of shape (T, C, 2*M).

    Returns:
        torch.Tensor: Output complex tensor of shape (T, C, M).
    """
    if not torch.is_floating_point(arr_real):
        raise TypeError("Input tensor must be a real float type.")
    if arr_real.dim() != 3:
        raise ValueError(f"Input tensor must be 3D (T, C, 2*M), but has "
                         f"{arr_real.dim()} dimensions.")
    
    # Ensure the last dimension is even
    if arr_real.shape[-1] % 2 != 0:
        raise ValueError("The last dimension of the real tensor must be even.")

    # Get the original number of channels M by dividing the last dim by 2.
    M_orig = arr_real.shape[-1] // 2
    
    # Split the tensor back into real and imaginary parts.
    real_part = arr_real[..., :M_orig]
    imag_part = arr_real[..., M_orig:]
    
    # Combine them into a complex number.
    return torch.complex(real_part, imag_part)


def compute_stats_with_healpy_tools(root_path: str, folder_list: list, out_stats_path: str) -> dict:
    """
    Computes normalization statistics by explicitly using healpy tools to interpret
    the coefficient array structure, ensuring correctness and building confidence.
    """

    def _norm_and_collect_explicit(arr: np.ndarray) -> dict:
        """
        Compute μ/σ for each individual (l,m) coefficient over an array of 
        shape (N, C, M).

        This function explicitly uses healpy functions to derive the structure.
        """
        # N: number of samples, C: number of alm coefficients, M: number of maps
        N, C, M = arr.shape
        
        # 1. Deduce Lmax FROM the size of the coefficient dimension (C) using healpy.
        # This is the inverse of hp.Alm.getsize().
        lmax = hp.Alm.getlmax(C)
        
        # Add a check for confidence. If C is not a valid size for a healpy array,
        # getlmax returns -1.
        if lmax == -1:
            raise ValueError(
                f"The number of coefficients C={C} is not a valid size "
                "for a standard healpy spherical harmonic array."
            )
        
        print(f"    Input array has C={C} coefficients, correctly interpreted as Lmax={lmax}.")

        # 2. Get the (l, m) values for EACH index `i` from 0 to C-1.
        # These tables map an index in the 1D array to its harmonic degree/order.
        l_vals, m_vals = hp.Alm.getlm(lmax=lmax)

        stats = {}
        for m_id in range(M):
            print(f"    Processing map ID {m_id+1}/{M}...")
            # We will store the results in simple 1D arrays of shape (C,)
            # that match the healpy ordering.
            mu_array = np.zeros(C, dtype=np.complex64)
            sigma_array = np.zeros(C, dtype=np.float32)

            # 3. Explicitly loop through every single coefficient from i = 0 to C-1.
            # This demonstrates we are handling the m-major order correctly.
            for i in range(C):
                # For demonstration, we can see which (l,m) pair this index `i` is.
                l, m = l_vals[i], m_vals[i]
                
                # Get all N samples for this single coefficient `i` (which is a_lm).
                # This slice has shape (N,).
                single_coeff_samples = arr[:, i, m_id]
                
                # Compute stats for this one coefficient.
                mu = single_coeff_samples.mean()
                sigma = single_coeff_samples.std()
                
                # Store the results in our arrays at the correct index `i`.
                mu_array[i] = mu
                sigma_array[i] = sigma

            # Store the complete, correctly ordered arrays for this m_id.
            # Add a small epsilon for numerical stability during normalization.
            stats[m_id] = (mu_array, sigma_array + 1e-9)
            
        return stats

    # --- The rest of the function remains the same ---

    radial_list, fg_list, bg_list = [], [], []
    print("Loading and collecting data...")
    for f in folder_list:
        p = os.path.join(root_path, f)
        if os.path.exists(os.path.join(p, "outer_sh_radial_latent.npy")):
            radial_list.append(np.load(os.path.join(p, "outer_sh_radial_latent.npy")))
        if os.path.exists(os.path.join(p, "outer_sh_phase_latent_fg_lmax22.npy")):
            fg_list.append(np.load(os.path.join(p, "outer_sh_phase_latent_fg_lmax22.npy")))
        if os.path.exists(os.path.join(p, "outer_sh_phase_latent_bg_lmax44.npy")):
            bg_list.append(np.load(os.path.join(p, "outer_sh_phase_latent_bg_lmax44.npy")))

    radial_all = np.concatenate(radial_list, axis=0)
    fg_all = np.concatenate(fg_list, axis=0)
    bg_all = np.concatenate(bg_list, axis=0)

    print("\nComputing statistics for 'radial'...")
    radial_stats = _norm_and_collect_explicit(radial_all)
    
    print("\nComputing statistics for 'phase_fg'...")
    fg_stats = _norm_and_collect_explicit(fg_all)
    
    print("\nComputing statistics for 'phase_bg'...")
    bg_stats = _norm_and_collect_explicit(bg_all)

    all_stats = {
        "radial": radial_stats,
        "phase_fg": fg_stats,
        "phase_bg": bg_stats
    }
    
    print(f"\nSaving correctly structured statistics to {out_stats_path}")
    with open(out_stats_path, "wb") as fp:
        pickle.dump(all_stats, fp)

    print("\nStats computation complete.")
    return all_stats



def compute_cfg_stats(root, folders, out_path):
    """Called **once** (just like SH stats)."""
    X = []
    for f in folders:
        cfg = json.load(open(os.path.join(root, f, "configuration.txt"))) \
              if f.endswith(".json") else parse_config_file(
                  os.path.join(root, f, "configuration.txt"))
        X.append([cfg[k] for k in CONFIG_KEYS])
    X   = np.asarray(X, np.float32)           # (N,P)
    μ   = X.mean(0)
    σ   = X.std (0) + 1e-6
    pickle.dump({"μ": μ, "σ": σ}, open(out_path, "wb"))
    return μ, σ