#!/usr/bin/env python
import numpy as np
from scipy.stats import qmc
import subprocess

# Define the 12 parameters with their bounds.
# Order:
# 0: k_bend_saturated: [4, 20]
# 1: k_bend_unsaturated: [0.1, 10]
# 2: k_bend_cholesterol: [1.0, 12.5]
# 3: w_c_default: [0.8, 2.5]
# 4: w_c_U-S: [0.8, 2.5]
# 5: w_c_U-C: [0.8, 2.5]
# 6: w_c_C-S: [0.8, 2.5]
# 7: Temperature: [0.6, 1.6]
# 8: outer_fraction_typeOne: [0.1, 0.9]
# 9: outer_fraction_typeTwo: [0.1, 0.9]  with constraint: (p8 + p9) <= 1.0
# 10: inner_fraction_typeThr: [0.1, 0.9]
# 11: inner_fraction_typeFour: [0.1, 0.9] with constraint: (p10 + p11) <= 1.0

bounds = np.array([
    [4.0, 20.0],
    [0.1, 10.0],
    [1.0, 12.5],
    [0.8, 2.5],
    [0.8, 2.5],
    [0.8, 2.5],
    [0.8, 2.5],
    [0.6, 1.6],
    [0.1, 0.9],
    [0.1, 0.9],
    [0.1, 0.9],
    [0.1, 0.9]
])

def generate_valid_samples(n_samples, dim, bounds):
    """
    Generates n_samples valid 12D samples using a Sobol sequence.
    For the outer (dims 8-9) and inner (dims 10-11) fractions,
    enforce that their sums are <= 1.0.
    """
    sampler = qmc.Sobol(d=dim, scramble=True)
    valid_samples = []
    # Generate candidates in batches until we have enough valid points.
    while len(valid_samples) < n_samples:
        # Use random_base2 with m=6 to generate 2^6 = 64 candidates per batch.
        candidates = sampler.random_base2(m=9)
        # Scale candidates to the given bounds.
        scaled = qmc.scale(candidates, bounds[:, 0], bounds[:, 1])
        for x in scaled:
            # Enforce that outer_typeOne + outer_typeTwo <= 1.0 and
            # inner_typeThr + inner_typeFour <= 1.0.
            if x[8] + x[9] <= 1.0 and x[10] + x[11] <= 1.0:
                valid_samples.append(x)
                print("valid samples: ",len(valid_samples))
                if len(valid_samples) >= n_samples:
                    break
    return np.array(valid_samples)

def main():
    n_samples = 256
    dim = 12
    samples = generate_valid_samples(n_samples, dim, bounds)
    
    # Save the samples to a CSV file for later review.
    header = ("k_bend_saturated,k_bend_unsaturated,k_bend_cholesterol,"
              "w_c_default,w_c_U-S,w_c_U-C,w_c_C-S,Temperature,"
              "outer_typeOne,outer_typeTwo,inner_typeThr,inner_typeFour")
    np.savetxt("parameter_samples.csv", samples, delimiter=",", header=header, comments='')
    
    for i, sample in enumerate(samples):
        # Pass the folder name as a positional argument (e.g., "config_1", "config_2", etc.)
        cmd = [
            "python", "run_simulation.py",
            f"config_{i+1}",
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
        print(f"Launching simulation {i+1}/{n_samples} with command:")
        print(" ".join(cmd))
        subprocess.call(cmd)

if __name__ == '__main__':
    main()