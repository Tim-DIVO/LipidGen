#!/usr/bin/env python3
"""
submit_jobs_nvt_long.py

Submit a Slurm job for every nvt_long.in under the project,
provided the folder:
  • has system_after_short_nvt.data
  • has a ‘stable’ marker file
  • does NOT yet have full_traj_long.lammpstrj
If a folder contains an 'unstable' marker, the entire folder is deleted.
"""

import logging
import shutil
import subprocess
from pathlib import Path

# ------------------------------------------------------------------
LAMMPS_EXE = "/home/timwehnes/lammps/build/lmp_2025"
SLURM_PART = "compute"
# ------------------------------------------------------------------

logging.basicConfig(
    filename="submit_jobs_nvt_long.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def discover_inputs():
    """Return sorted list of every **/nvt_long.in."""
    return sorted(Path(".").glob("**/nvt_long.in"))


def eligible(folder: Path) -> bool:
    """
    Folder qualifies if:
      1) short‑run data is present
      2) there's a 'stable' marker file
      3) long‑run output does NOT already exist
    """
    has_short_data = (folder / "system_after_short_nvt.data").is_file()
    has_stable     = (folder / "stable.txt").is_file() or (folder / "stable").is_file()
    already_done   = (folder / "full_traj_long.lammpstrj").is_file()

    if not has_short_data:
        logging.warning(f"SKIP {folder}: missing system_after_short_nvt.data")
    if not has_stable:
        logging.info(f"SKIP {folder}: not marked stable or marked unstable")
    if already_done:
        logging.info(f"SKIP {folder}: full_traj_long.lammpstrj already exists")

    return has_short_data and has_stable and not already_done


def make_and_submit(nvt_long: Path):
    folder = nvt_long.parent
    job_name    = f"{folder.name}_nvtL"
    script_path = folder / "job_nvt_long.sh"

    script_path.write_text(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition={SLURM_PART}
#SBATCH --time=02:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3GB
#SBATCH --account=innovation

module load 2024r1
module load openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "{folder}"
srun {LAMMPS_EXE} -in "{nvt_long.name}"
""")

    subprocess.run(["sbatch", str(script_path)], check=True)
    logging.info(f"Submitted {script_path}")


def main():
    inputs = discover_inputs()
    logging.info(f"Found {len(inputs)} nvt_long.in files")

    for nvt in inputs:
        folder = nvt.parent
        # If folder marked unstable, delete it entirely and skip
        if (folder / "unstable.txt").is_file() or (folder / "unstable").is_file():
            logging.info(f"Removing unstable folder {folder}")
            try:
                shutil.rmtree(folder)
                logging.info(f"Deleted folder {folder}")
            except Exception as e:
                logging.error(f"Error deleting {folder}: {e}")
            continue

        if eligible(folder):
            try:
                make_and_submit(nvt)
            except subprocess.CalledProcessError as e:
                logging.error(f"FAILED submitting {folder}: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()

