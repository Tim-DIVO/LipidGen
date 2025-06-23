#!/usr/bin/env python
import numpy as np
import subprocess

def main():
    # Define the range for w_c_default: from 0.6 to 1.8 in steps of 0.05.
    # This creates a total of 25 measurements.
    wc_default_values = np.arange(0.6, 1.8 + 0.05, 0.05)
    
    # Save the w_c_default values to a CSV file for reference.
    np.savetxt("wc_default_samples.csv", wc_default_values, delimiter=",", header="w_c_default", comments='')
    
    # Iterate over each w_c_default value and launch the simulation.
    for i, wc_default in enumerate(wc_default_values):
        cmd = [
            "python", "run_simulation.py",
            f"config_{i+1}",
            "--w_c_default", f"{wc_default:.4f}"
        ]
        print(f"Launching simulation {i+1} with w_c_default = {wc_default:.4f}")
        print(" ".join(cmd))
        subprocess.call(cmd)

if __name__ == '__main__':
    main()
