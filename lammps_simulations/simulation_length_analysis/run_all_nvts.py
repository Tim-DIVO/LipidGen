#!/usr/bin/env python
import os
import glob
import subprocess
import logging

logging.basicConfig(
    filename="submit_jobs_nvt.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    # 1) Find all 'nvt_*.in' in the current directory
    nvt_files = sorted(glob.glob("nvt_*.in"))
    logging.info(f"Found {len(nvt_files)} nvt files: {nvt_files}")

    # 2) Absolute path to your LAMMPS binary
    lammps_exe = "/home/timwehnes/lammps/build/lmp_2025"

    for f in nvt_files:
        base = os.path.splitext(f)[0]  # e.g. nvt_chol0.20_equal
        job_name = base
        script_filename = f"job_{base}.sh"

        script_contents = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=innovation

module load 2024r1
module load openmpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun {lammps_exe} -in {f}
"""

        with open(script_filename, 'w') as out:
            out.write(script_contents)

        try:
            sbatch_cmd = ["sbatch", script_filename]
            subprocess.run(sbatch_cmd, check=True)
            logging.info(f"Submitted {script_filename} for {f}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed submitting {script_filename} for {f}: {e}")

    logging.info("All NVT jobs submitted.")

if __name__ == "__main__":
    main()
