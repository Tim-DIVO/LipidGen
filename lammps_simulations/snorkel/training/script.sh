#!/bin/bash

#SBATCH --job-name=lammps_1
#SBATCH --partition=compute
#SBATCH --time=20:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=innovation

# Load modules:
module load 2024r1
module load openmpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python run_auto.py