#!/usr/bin/env python
"""
sobol_pipeline.py

Generates Sobol samples for 12 parameters with two sum constraints
(p8+p9<=1, p10+p11<=1) via the folding trick, scales to real bounds,
writes to CSV, and launches simulations.
Supports reproducible seeds and incremental sampling via a start index.
"""

import numpy as np
from scipy.stats import qmc
import subprocess
import argparse

# Default seed for reproducibility
SEED = 42

# Define parameter bounds: [min, max] for each of the 12 parameters
bounds = np.array([
    [4.0, 20.0],   # k_bend_saturated
    [0.1, 10.0],   # k_bend_unsaturated
    [1.0, 12.5],   # k_bend_cholesterol
    [0.8, 2.5],    # w_c_default
    [0.8, 2.5],    # w_c_U-S
    [0.8, 2.5],    # w_c_U-C
    [0.8, 2.5],    # w_c_C-S
    [0.6, 1.6],    # Temperature
    [0.1, 0.9],    # outer_fraction_typeOne
    [0.1, 0.9],    # outer_fraction_typeTwo  (with p8+p9<=1)
    [0.1, 0.9],    # inner_fraction_typeThr
    [0.1, 0.9],    # inner_fraction_typeFour  (with p10+p11<=1)
])

def generate_samples(n_samples, bounds, start_index=0, seed=None):
    """
    Generate `n_samples` Sobol points in [0,1]^12 starting at `start_index`,
    fold the (8,9) and (10,11) pairs into their triangles, then scale.
    Uses `seed` for scramble reproducibility.
    """
    dim = bounds.shape[0]
    sob = qmc.Sobol(d=dim, scramble=True, seed=seed)

    # Skip ahead to `start_index`
    if start_index > 0:
        try:
            sob.fast_forward(start_index)
        except AttributeError:
            # SciPy <1.8 fallback: generate and discard
            _ = sob.random(start_index)

    U = sob.random(n_samples)  # shape (n_samples, 12)

    # Fold outer fractions (dims 8,9) into u+v<=1 triangle
    mask_outer = (U[:, 8] + U[:, 9] > 1)
    U[mask_outer, 8:10] = 1 - U[mask_outer, 8:10]

    # Fold inner fractions (dims 10,11) into u+v<=1 triangle
    mask_inner = (U[:, 10] + U[:, 11] > 1)
    U[mask_inner, 10:12] = 1 - U[mask_inner, 10:12]

    # Scale all dims into provided bounds
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + (hi - lo) * U

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sobol sampling with sum constraints")
    parser.add_argument("--n-samples", type=int, default=256,
                        help="Number of new samples to generate")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Index of first sample to generate (0-based)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for Sobol scramble")
    args = parser.parse_args()

    # Generate samples
    samples = generate_samples(args.n_samples, bounds,
                               start_index=args.start_index,
                               seed=args.seed)

    # Save to CSV
    out_csv = f"parameter_samples_{args.start_index}_" + f"{args.start_index + args.n_samples}.csv"
    header = (
        "k_bend_saturated,k_bend_unsaturated,k_bend_cholesterol,"
        "w_c_default,w_c_U-S,w_c_U-C,w_c_C-S,Temperature,"
        "outer_typeOne,outer_typeTwo,inner_typeThr,inner_typeFour"
    )
    np.savetxt(out_csv, samples, delimiter=",",
               header=header, comments='')
    print(f"Saved {args.n_samples} samples to {out_csv}")

    # Launch simulations
    for i, sample in enumerate(samples, start=args.start_index + 1):
        cmd = ["python", "2_data_creation.py", f"config_{i}"]
        cmd += [
            "--k_bend_saturated", f"{sample[0]:.4f}",
            "--k_bend_unsaturated", f"{sample[1]:.4f}",
            "--k_bend_cholesterol", f"{sample[2]:.4f}",
            "--w_c_default", f"{sample[3]:.4f}",
            "--w_c_U_S", f"{sample[4]:.4f}",
            "--w_c_U_C", f"{sample[5]:.4f}",
            "--w_c_C_S", f"{sample[6]:.4f}",
            "--Temperature", f"{sample[7]:.4f}",
            "--outer_typeOne", f"{sample[8]:.4f}",
            "--outer_typeTwo", f"{sample[9]:.4f}",
            "--inner_typeThr", f"{sample[10]:.4f}",
            "--inner_typeFour", f"{sample[11]:.4f}"
        ]
        print(f"Launching simulation {i}")
        subprocess.call(cmd)
