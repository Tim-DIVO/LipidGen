#!/usr/bin/env python
import os
import glob
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    filename="submit_jobs_min.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------------------------------------------------
# absolute path to the LAMMPS binary on your cluster
LAMMPS_EXE = "/home/timwehnes/lammps/build/lmp_2025"
# ----------------------------------------------------------------------

def discover_min_inputs():
    """
    Return a sorted list of Path objects for every
      <anything>/min*.in
    below the current working directory.
    """
    # **/ is the recursive wildcard; recursive=True is required
    pattern = "**/min*.in"
    return sorted(Path(".").glob(pattern))

def submit_job(min_in: Path):
    """
    Compose and submit a Slurm script that runs LAMMPS on `min_in`.
    """
    folder      = min_in.parent            # e.g. config_12
    input_name  = min_in.name              # e.g. min.in
    job_name    = f"{folder.name}_min"     # Slurm job‑name
    script_path = folder / f"job_{input_name}.sh"

    script_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out        # stdout  (e.g. config_7_min_123456.out)
#SBATCH --error=%x_%j.err         # stderr  (uncomment if you want a separate file)
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=innovation

module load 2024r1
module load openmpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "{folder}"          # run inside the config folder
srun {LAMMPS_EXE} -in "{input_name}"
"""

    script_path.write_text(script_contents)

    try:
        subprocess.run(["sbatch", str(script_path)], check=True)
        logging.info(f"Submitted {script_path} for {min_in}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED submitting {script_path} for {min_in}: {e}")

def main():
    min_inputs = discover_min_inputs()
    logging.info(f"Found {len(min_inputs)} minimization inputs.")
    for min_in in min_inputs:
        submit_job(min_in)
    logging.info("Done submitting all minimization jobs.")

if __name__ == "__main__":
    main()
