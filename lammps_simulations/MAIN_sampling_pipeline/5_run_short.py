#!/usr/bin/env python
"""
submit_jobs_nvt_short.py

Recursively locate every nvt_short.in below the current directory,
verify that its folder
  • contains system_after_min.data
  • does NOT yet contain system_after_short_nvt.data
and submit a Slurm job that runs LAMMPS on it.
"""

import subprocess
import logging
from pathlib import Path

# ------------------------------------------------------------------
LAMMPS_EXE = "/home/timwehnes/lammps/build/lmp_2025"   # <-- adjust if needed
SLURM_PART = "compute"
# ------------------------------------------------------------------

logging.basicConfig(
    filename="submit_jobs_nvt_short.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def find_inputs():
    """Return sorted list of Path objects matching **/nvt_short.in."""
    return sorted(Path(".").glob("**/nvt_short.in"))

def ready_for_short_nvt(folder: Path) -> bool:
    """
    True if folder has system_after_min.data but NOT system_after_short_nvt.data.
    """
    has_min   = (folder / "system_after_min.data").is_file()
    done_already = (folder / "system_after_short_nvt.data").is_file()
    if not has_min:
        logging.warning(f"SKIP {folder}: system_after_min.data missing")
    if done_already:
        logging.info(f"SKIP {folder}: short NVT already completed")
    return has_min and not done_already

def submit(nvt_in: Path):
    folder = nvt_in.parent
    job_name    = f"{folder.name}_nvtS"
    script_path = folder / "job_nvt_short.sh"

    script_path.write_text(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=%x_%j.out        # stdout  (e.g. config_7_min_123456.out)
#SBATCH --error=%x_%j.err         # stderr  (uncomment if you want a separate file)
#SBATCH --partition={SLURM_PART}
#SBATCH --time=02:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=innovation

module load 2024r1
module load openmpi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "{folder}"
srun {LAMMPS_EXE} -in "{nvt_in.name}"
""")

    subprocess.run(["sbatch", str(script_path)], check=True)
    logging.info(f"Submitted {script_path}")

def main():
    inputs = find_inputs()
    logging.info(f"Discovered {len(inputs)} nvt_short.in files")
    for nvt in inputs:
        if ready_for_short_nvt(nvt.parent):
            try:
                submit(nvt)
            except subprocess.CalledProcessError as e:
                logging.error(f"FAILED submitting {nvt.parent}: {e}")
    logging.info("Finished processing all nvt‑short jobs")

if __name__ == "__main__":
    main()

