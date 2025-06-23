#!/usr/bin/env python
import os
import glob
import subprocess
import logging

logging.basicConfig(
    filename="submit_jobs_min.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    # 1) Find all 'min_*.in' in the current directory
    min_files = sorted(glob.glob("min_*.in"))
    logging.info(f"Found {len(min_files)} min files: {min_files}")

    # 2) Absolute path to your LAMMPS binary
    lammps_exe = "/home/timwehnes/lammps/build/lmp_2025"

    for f in min_files:
        # We'll strip off ".in" to create a job script name
        base = os.path.splitext(f)[0]  # e.g. min_chol0.20_equal
        job_name = base
        script_filename = f"job_{base}.sh"

        # Slurm job script
        script_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=innovation

module load 2024r1
module load openmpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run LAMMPS minimization right here
srun {lammps_exe} -in {f}
"""

        # Write the script
        with open(script_filename, 'w') as out:
            out.write(script_contents)

        # Submit the script
        try:
            sbatch_cmd = ["sbatch", script_filename]
            subprocess.run(sbatch_cmd, check=True)
            logging.info(f"Submitted {script_filename} for {f}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed submitting {script_filename} for {f}: {e}")

    logging.info("All min jobs submitted.")

if __name__ == "__main__":
    main()
