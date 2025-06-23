import os
import pickle
import numpy as np
import torch
import math
import json
from utils import complex_to_real_channel_stack, cfg2vec, parse_config_file
import healpy as hp

def unnorm_healpix_alms_final(arr_norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Reverses the normalization process for healpy alm coefficients.
    This function adheres to the healpy m-major vector ordering.

    Args:
        arr_norm (np.ndarray): Normalized input array of shape (T, C, M).
                               MUST have a complex dtype.
        stats (dict):          A dictionary with the structure:
                               {m_id: (mu_array, sigma_array)}
                               where mu_array is complex (C,) and sigma_array is real (C,).

    Returns:
        np.ndarray: Un-normalized array in its original scale, 
                    shape (T, C, M), dtype complex64.
    
    Raises:
        TypeError: If the input array 'arr_norm' is not complex.
        ValueError: If the coefficient dimension 'C' is not a valid healpy size.
    """
    # 1. Validate inputs: Ensure the array is complex.
    if not np.iscomplexobj(arr_norm):
        raise TypeError(
            f"Input array 'arr_norm' must be complex, but its dtype is {arr_norm.dtype}."
        )

    T, C, M = arr_norm.shape

    # 2. Compute Lmax FROM the size of the coefficient dimension (C) using healpy.
    lmax = hp.Alm.getlmax(C)
    if lmax == -1:
        raise ValueError(
            f"The number of coefficients C={C} is not a valid size for a "
            "standard healpy spherical harmonic array."
        )
    #print(f"Un-normalizing array with C={C} (Lmax={lmax}) and {M} channels.")

    # 3. Prepare the output array. We can use np.empty_like to match shape and dtype.
    out = np.empty_like(arr_norm)

    # 4. Loop through each map/channel in the M dimension.
    for m_id in range(M):
        # Retrieve the pre-computed statistics for this map ID.
        # These are 1D arrays of shape (C,) in the correct m-major order.
        if m_id not in stats:
            raise KeyError(f"Statistics for m_id={m_id} not found in the stats dictionary.")
        
        mu, sigma = stats[m_id]

        # Select the slice of the normalized input array for this map ID.
        norm_slice = arr_norm[..., m_id] # Shape (T, C)
        
        # 5. Apply the inverse normalization using NumPy broadcasting.
        # The operation is the reverse: y' = y * σ + μ
        # NumPy correctly broadcasts the (C,) arrays over the (T, C) array.
        out[..., m_id] = norm_slice * sigma + mu
            
    return out

def merge_coeffs(x):               # x: [B,T,C,M] complex
    B, T, C, M = x.shape
    return x.reshape(B, T, C*M)    # -> [B,T, C·M] complex


def apply_healpix_normalization_final(arr: np.ndarray, stats: dict) -> np.ndarray:
    """
    Given a raw data array and correctly computed stats, apply normalization.
    This function adheres to the healpy m-major vector ordering.

    Args:
        arr (np.ndarray): Input array of shape (T, C, M).
                          MUST have a complex dtype.
        stats (dict):     A dictionary with the structure:
                          {m_id: (mu_array, sigma_array)}
                          where mu_array is complex (C,) and sigma_array is real (C,).

    Returns:
        np.ndarray: Normalized array, shape (T, C, M), dtype complex64.
    
    Raises:
        TypeError: If the input array 'arr' is not complex.
        ValueError: If the coefficient dimension 'C' is not a valid healpy size.
    """
    # 1. Validate inputs: Ensure the array is complex as required.
    if not np.iscomplexobj(arr):
        raise TypeError(
            f"Input array 'arr' must be complex, but its dtype is {arr.dtype}."
        )

    T, C, M = arr.shape

    # 2. Compute Lmax FROM the size of the coefficient dimension (C) using healpy.
    lmax = hp.Alm.getlmax(C)
    if lmax == -1:
        raise ValueError(
            f"The number of coefficients C={C} is not a valid size for a "
            "standard healpy spherical harmonic array."
        )
    #print(f"Normalizing array with C={C} (Lmax={lmax}) and {M} channels.")

    # 3. Prepare the output array with the correct complex dtype.
    out = np.empty((T, C, M), dtype=np.complex64)

    # 4. Loop through each map/channel in the M dimension.
    for m_id in range(M):
        # Retrieve the pre-computed statistics for this map ID.
        # mu is a complex 1D array of shape (C,).
        # sigma is a real 1D array of shape (C,).
        # Their ordering is the correct m-major ordering.
        if m_id not in stats:
            raise KeyError(f"Statistics for m_id={m_id} not found in the stats dictionary.")
        
        mu, sigma = stats[m_id]

        # Select the slice of the input array for this map ID. Shape: (T, C)
        data_slice = arr[..., m_id]
        
        # 5. Apply normalization using NumPy broadcasting.
        # The operation (T, C) / (C,) is handled element-wise by NumPy
        # along the T dimension, which is exactly what we want.
        # No complex indexing or manual slicing is needed.
        out[..., m_id] = (data_slice - mu) / sigma
            
    return out

def normalize_folder(root_path: str, folder: str, stats: dict) -> dict:
    """
    For a given folder, load raw latents and apply normalization using precomputed stats.
    Returns a dict with normalized arrays under keys "radial", "fg", "bg".
    """
    p = os.path.join(root_path, folder)
    radial_raw = np.load(os.path.join(p, "outer_sh_radial_latent.npy"))       # (T, 120, 1)
    fg_raw     = np.load(os.path.join(p, "outer_sh_phase_latent_fg_lmax22.npy"))  # (T, 276, 9)
    bg_raw     = np.load(os.path.join(p, "outer_sh_phase_latent_bg_lmax44.npy"))  # (T, 1035, 3)

    return {
        "radial": apply_healpix_normalization_final(radial_raw, stats["radial"]),
        "fg":     apply_healpix_normalization_final(fg_raw,     stats["phase_fg"]),
        "bg":     apply_healpix_normalization_final(bg_raw,     stats["phase_bg"])
    }


# ---------- helper to build index map ONCE ----------
def _make_maps(Lmax: int):
    """
    Returns two python lists:
      idx_posm -> idx_real   (length C_pos)
      idx_real -> (m, sign)  (length C_real)   sign = +1 if maps to Re, −1 if to Im
    The “real” order is [0, +1, -1, +2, -2, …, +ℓ, -ℓ] per band.
    """
    pos2real, real2meta = [], []
    r = 0
    for ℓ in range(Lmax + 1):
        # m = 0
        pos2real.append(r);  real2meta.append((0, +1));  r += 1
        for m in range(1, ℓ + 1):
            pos2real.append(r)     # +m  -> Im
            real2meta.append(( m, -1))
            r += 1
            pos2real.append(r)     # -m  -> Re
            real2meta.append((-m, +1))
            r += 1
    return pos2real, real2meta



class HealpixSequenceDataset:
    """
    A dataset that:
      - finds all subfolders under `root_path` whose names start with "configf_" (or you can pass a list of folder names),
      - for each index i, it:
          1. reads that folder’s 'configuration.txt' → a small dict of floats,
          2. loads the three Healpix‐latent .npy files (radial, fg, bg),
          3. applies pre‐computed normalization via `apply_healpix_normalization(...)`,
          4. optionally applies `transform(...)` to (radial_norm, fg_norm, bg_norm),
          5. returns a dict:
             {
               "radial":   radial_norm_array,   # shape (T, C_radial, 1)
               "fg":       fg_norm_array,       # shape (T, C_fg, M_fg)
               "bg":       bg_norm_array,       # shape (T, C_bg, M_bg)
               "config":   config_dict,         # e.g. {'k_bend_saturated': 19.21, …, 'folder': 'configf_0001'}
               "folder":   folder_name_string   # e.g. 'configf_0001'
             }
    """

    def __init__(
        self,
        root_path: str,
        folder_list: list = None,
        stats_path: str = "sh_norm_stats.pkl",
        transform: callable = None,
        discover_pattern: str = "configf_",
        cfg_stats_path="config_norm.pkl"
    ):
        """
        Args:
          root_path:      top‐level directory where all "configf_*" subfolders live.
          folder_list:    (optional) a list of subfolder names. If None, we auto‐discover any subdir whose name starts with `discover_pattern`.
          stats_path:     path to the pickled { "radial":…, "phase_fg":…, "phase_bg":… } dict.
          transform:      an optional function that takes (radial_norm, fg_norm, bg_norm, config) 
                          and returns a modified tuple; e.g. for data‐augmentation.  If None, no augmentation.
          discover_pattern: a string; any folder whose name starts with this is treated as a data folder.
        """
        self.root_path = root_path
        self.transform = transform

        self.cfg_stats = pickle.load(open(cfg_stats_path, "rb"))


        # 1) Load the stats once
        with open(stats_path, "rb") as fp:
            self.stats = pickle.load(fp)
        #  self.stats["radial"], self.stats["phase_fg"], self.stats["phase_bg"]

        # 2) If folder_list not provided, auto‐discover:
        if folder_list is None:
            all_entries = os.listdir(root_path)
            # keep only directories that start with e.g. "configf_"
            self.folder_list = [
                fname for fname in all_entries
                if fname.startswith(discover_pattern)
                and os.path.isdir(os.path.join(root_path, fname))
            ]
            self.folder_list.sort()  # sort for reproducibility
        else:
            self.folder_list = sorted(folder_list)

        # 3) Sanity: ensure each folder has the three expected .npy files + configuration.txt
        for f in self.folder_list:
            p = os.path.join(root_path, f)
            if not os.path.isdir(p):
                raise ValueError(f"Expected folder {p} to exist.")
            for fn in (
                "outer_sh_radial_latent.npy",
                "outer_sh_phase_latent_fg_lmax22.npy",
                "outer_sh_phase_latent_bg_lmax44.npy",
                "configuration.txt",
            ):
                full = os.path.join(p, fn)
                if not os.path.exists(full):
                    raise FileNotFoundError(f"Missing {fn} under {p}")

    def __len__(self) -> int:
        return len(self.folder_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with keys:
          'radial', 'fg', 'bg', 'config', 'folder'
          - radial: np.ndarray shape (T, C_radial, 1), dtype float32/complex64
          - fg:     np.ndarray shape (T, C_fg, M_fg)
          - bg:     np.ndarray shape (T, C_bg, M_bg)
          - config: dict of floats + 'folder'
          - folder: string (same as config['folder'])
        """
        folder_name = self.folder_list[idx]
        p = os.path.join(self.root_path, folder_name)

        # 1) parse configuration.txt
        #config_path = os.path.join(p, "configuration.txt")
        #config_dict = parse_config_file(config_path)

        #
        cfg = parse_config_file(os.path.join(p, "configuration.txt"))
        cond_vec = cfg2vec(cfg, μσ=(self.cfg_stats["μ"], self.cfg_stats["σ"])).to(torch.float32)  # (P,)

        # 2) load raw healpix latent .npy arrays
        radial_raw = np.load(os.path.join(p, "outer_sh_radial_latent.npy"))
        fg_raw     = np.load(os.path.join(p, "outer_sh_phase_latent_fg_lmax22.npy"))
        bg_raw     = np.load(os.path.join(p, "outer_sh_phase_latent_bg_lmax44.npy"))

        # 3) normalize each with pre‐computed stats
        radial_norm = apply_healpix_normalization_final(radial_raw, self.stats["radial"])
        fg_norm     = apply_healpix_normalization_final(fg_raw,     self.stats["phase_fg"])
        bg_norm     = apply_healpix_normalization_final(bg_raw,     self.stats["phase_bg"])

        # 4) optionally apply a user‐provided transform (e.g. augmentation)
        if self.transform is not None:
            # Pass all three arrays plus config_dict into transform, let it return a new tuple
            # e.g. transform might be: def aug(radial, fg, bg, config): … ; return (radial_aug, fg_aug, bg_aug)
            radial_norm, fg_norm, bg_norm = self.transform(
                radial_norm, fg_norm, bg_norm
            )

        # STEP 4: Convert to PyTorch Tensors and Apply Basis Transformation.
        
        # Convert numpy arrays to PyTorch complex tensors first.
        radial_cplx = torch.from_numpy(radial_norm).to(torch.complex64)
        fg_cplx     = torch.from_numpy(fg_norm).to(torch.complex64)
        bg_cplx     = torch.from_numpy(bg_norm).to(torch.complex64)

        #print("Converting radial...")
        radial_real = complex_to_real_channel_stack(radial_cplx)

        #print("Converting fg...")
        fg_real = complex_to_real_channel_stack(fg_cplx)

        #print("Converting bg...")
        bg_real = complex_to_real_channel_stack(bg_cplx)

        return {
            "radial": radial_real,  # float  (T,C_r,1)
            "fg"    : fg_real,      # float  (T,C_f,9)
            "bg"    : bg_real,      # float  (T,C_b,3)
            "cond"  : cond_vec,     # float32    (P,)
            "folder": folder_name
        }


def collate_complex(batch):
    """
    batch = list of dicts produced by dataset.__getitem__.
            Each dict has keys: radial, fg, bg (complex), cond (float),
            folder (str).

    Returns one dict where tensors are stacked along dim-0, e.g.
        radial : (B,T,C_r,1) complex64
        cond   : (B,P)       float32
        folder : list[str]   length B
    """
    out = {}
    for key in ("radial", "fg", "bg"):
        out[key] = torch.stack([sample[key].clone()           # clone() → make storage resizable
                                for sample in batch], dim=0)  # (B, …)
    out["cond"]   = torch.stack([sample["cond"] for sample in batch], 0)  # float32
    out["folder"] = [sample["folder"] for sample in batch]                # python list
    return out

