#!/usr/bin/env python
import os
import glob
import subprocess

def run_simulation_in_folder(folder, input_file):
    """
    Change into the given folder, run the LAMMPS simulation using the specified
    input file, then return to the parent directory.
    """
    original_dir = os.getcwd()
    os.chdir(folder)
    print(f"Running {input_file} in {folder} ...")
    # Replace 'lmp_serial' with your LAMMPS executable (e.g., lmp_mpi) as needed.
    #cmd = ["mpirun -np 2 lmp_mpi", "-in", input_file]
    cmd = ["mpirun", "-np", "5", "lmp_mpi", "-in", input_file]

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running {input_file} in {folder}: {e}")
    os.chdir(original_dir)

def main():
    # Find all config folders (e.g., config_1, config_2, ...)
    config_folders = sorted(glob.glob("config_*"), key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    #print("Launching minimization runs in all config folders...")
    #for folder in config_folders:
    #    run_simulation_in_folder(folder, "min.in")
    
    print("All minimizations complete. Launching NVT runs...")
    for folder in config_folders:
        run_simulation_in_folder(folder, "nvt.in")
    
    print("All NVT simulations launched.")

if __name__ == "__main__":
    main()
