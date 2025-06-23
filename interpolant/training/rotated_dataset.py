# rotated_dataset.py
import torch, random, math, hashlib, numpy as np
from torch.utils.data import Dataset
from rotate_alm_helper import rotate_trajectory            # rotates complex Alm
from dataset import HealpixSequenceDataset
from utils import complex_to_real_channel_stack, real_to_complex_channel_stack

class RotatedHealpixDataset(Dataset):
    """
    For each underlying sample returns (k_rot + 1) views:
        aug_id = 0    original orientation
        aug_id = 1..k random SO(3) rotations PER EPOCH
    """
    def __init__(self, base_ds: HealpixSequenceDataset, k_rot: int = 2):
        self.base   = base_ds
        self.k_rot  = int(k_rot)
        self.index  = [(i, j) for i in range(len(base_ds))
                               for j in range(k_rot + 1)]
        self.epoch  = 0        # will be set from trainer

    # ---------------- utilities ----------------
    def _random_angles(self, folder_name: str, aug_id: int):
        """Generate epoch-dependent but reproducible angles."""
        mix = f"{folder_name}_{self.epoch}_{aug_id}"
        seed = int(hashlib.sha1(mix.encode()).hexdigest(), 16) % 2**32
        rng  = random.Random(seed)
        psi   = rng.uniform(0, 2*math.pi)
        theta = math.acos(1 - 2*rng.random())              # uniform on sphere
        phi   = rng.uniform(0, 2*math.pi)
        return psi, theta, phi

    # -------- PyTorch hooks --------------------
    def set_epoch(self, epoch:int):
        """Call from training loop so angles change each epoch."""
        self.epoch = epoch

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        base_idx, aug_id = self.index[idx]
        sample = self.base[base_idx]         # dict with float32 real tensors

        if aug_id == 0:
            return sample                    # original

        # 1) bring back to complex pos-m
        rad_c = real_to_complex_channel_stack(sample["radial"])   # (B, C_pos, 1)
        fg_c  = real_to_complex_channel_stack(sample["fg"])
        bg_c  = real_to_complex_channel_stack(sample["bg"])
       
        # 2) rotate
        psi,theta,phi = self._random_angles(sample["folder"], aug_id)
        rad_rot_c = rotate_trajectory(rad_c.numpy(), psi,theta,phi)
        fg_rot_c  = rotate_trajectory(fg_c.numpy() , psi,theta,phi)
        bg_rot_c  = rotate_trajectory(bg_c.numpy() , psi,theta,phi)

        # 3) convert back to real basis float32
        radial_real = complex_to_real_channel_stack(torch.from_numpy(rad_rot_c))
        fg_real     = complex_to_real_channel_stack(torch.from_numpy(fg_rot_c))
        bg_real     = complex_to_real_channel_stack(torch.from_numpy(bg_rot_c))

        sample = sample.copy()   # avoid side-effects
        sample["radial"] = radial_real.float()
        sample["fg"]     = fg_real.float()
        sample["bg"]     = bg_real.float()
        return sample
