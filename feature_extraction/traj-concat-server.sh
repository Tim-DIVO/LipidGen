#!/bin/bash
#SBATCH --job-name=concat
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=research-as-bn
#SBATCH --output=concat.%j.out
#SBATCH --error=concat.%j.err


# 1) load the Conda module (if provided)
module load 2024r1

# 4) run your script
srun python traj-concat-server.py
