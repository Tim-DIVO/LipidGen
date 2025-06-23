#!/usr/bin/env python
import argparse
import os
import subprocess
import shutil
import time

# ----- Updated MINIMIZATION SCRIPT with placeholders -----
MINIMIZATION_SCRIPT = r"""# ----------------- Init Section -----------------
# system.in.init

units           lj
dimension       3
atom_style      full

bond_style      hybrid fene harmonic 
angle_style     none

pair_style cosine/squared 3.0

special_bonds   lj 0.0 1.0 1.0

neighbor        0.4 bin
neigh_modify    every 2 delay 0 check yes

read_data bilayer.data

# ---------------- Settings Section -----------------
bond_style hybrid fene harmonic 

# adjacency for all species
bond_coeff 1 fene     30.0 1.5 1.0 1.0

# bridging for unsat - HALF K-VALUE FROM LITERATURE
bond_coeff 2 harmonic  2.5 4.0   # e.g. k=5, r0=4.0

# bridging for sat - HALF K-VALUE FROM LITERATURE
bond_coeff 3 harmonic 12.5 4.0   # e.g. k=25, r0=4.0

# bridging for chol - HALF K-VALUE FROM LITERATURE
bond_coeff 4 harmonic 5.0 4.0   # e.g. k=10, r0=4.0

pair_style cosine/squared 3.0

# (A) Head–anything => purely repulsive WCA
pair_coeff 1 1 1.0  1.0663  1.0663 wca
pair_coeff 1 2 1.0  1.0934  1.0934 wca
pair_coeff 1 3 1.0  1.0934  1.0934 wca
pair_coeff 1 4 1.0  1.0934  1.0934 wca


# ----------------------------------------------------------------
# (B) Cross-leaflet mid–mid => purely repulsive WCA
# ----------------------------------------------------------------
pair_coeff 3 4 1.0 1.1225 1.1225 wca


# (C2) unsat–unsat => 
    #tails
pair_coeff 2 2 1.0  1.1225  {w_c_default_term} wca
    #outer layer
pair_coeff 2 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 3 3 1.0  1.1225  {w_c_default_term} wca
    #inner layer
pair_coeff 2 4 1.0  1.1225 {w_c_default_term} wca
pair_coeff 4 4 1.0  1.1225 {w_c_default_term} wca


comm_modify cutoff 7.0
dump            1 all custom 50 traj_min.lammpstrj id mol type x y z ix iy iz
thermo_style    custom step pe etotal vol epair ebond eangle
thermo          40

minimize 1.0e-7 1.0e-9 10000 300000

write_data system_after_min.data
"""

# ----- Updated NVT SCRIPT with placeholders -----
NVT_SCRIPT = r"""# ----------------- Init Section -----------------
units           lj
dimension       3
atom_style      full

bond_style      hybrid fene harmonic 
angle_style     none

pair_style cosine/squared 3.0

special_bonds   lj 0.0 1.0 1.0

neighbor        0.4 bin
neigh_modify    every 2 delay 0 check yes

read_data system_after_min.data

bond_style hybrid fene harmonic 
bond_coeff 1 fene     30.0 1.5 1.0 1.0
# bridging for unsat - HALF K-VALUE FROM LITERATURE
bond_coeff 2 harmonic  2.5 4.0   # e.g. k=5, r0=4.0

# bridging for sat - HALF K-VALUE FROM LITERATURE
bond_coeff 3 harmonic 12.5 4.0   # e.g. k=25, r0=4.0

# bridging for chol - HALF K-VALUE FROM LITERATURE
bond_coeff 4 harmonic 5.0 4.0   # e.g. k=10, r0=4.0

pair_style cosine/squared 3.0

# (A) Head–anything => purely repulsive WCA
pair_coeff 1 1 1.0  1.0663  1.0663 wca
pair_coeff 1 2 1.0  1.0934  1.0934 wca
pair_coeff 1 3 1.0  1.0934  1.0934 wca
pair_coeff 1 4 1.0  1.0934  1.0934 wca


# ----------------------------------------------------------------
# (B) Cross-leaflet mid–mid => purely repulsive WCA
# ----------------------------------------------------------------
pair_coeff 3 4 1.0 1.1225 1.1225 wca


# (C2) unsat–unsat => 
    #tails
pair_coeff 2 2 1.0  1.1225  {w_c_default_term} wca
    #outer layer
pair_coeff 2 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 3 3 1.0  1.1225  {w_c_default_term} wca
    #inner layer
pair_coeff 2 4 1.0  1.1225 {w_c_default_term} wca
pair_coeff 4 4 1.0  1.1225 {w_c_default_term} wca

comm_modify cutoff 7.0
# ---- Normal Coordinate Trajectory Dump ----
# Dump atom id, type, and positions every 1000 steps.
dump           1 all custom 300 full_traj.lammpstrj id type x y z
dump_modify    1 sort id


# ---- Define Global Variables to be Dumped ----
variable ke   equal ke         # kinetic energy
variable pe   equal pe         # potential energy
variable tot  equal etotal     # total energy
variable temp equal temp       # temperature

# Define a custom variable for the current simulation step
variable simstep equal step

fix global_print all print 1000 "Step: ${{simstep}}  KE: ${{ke}}  PE: ${{pe}}  Tot: ${{tot}}  Temp: ${{temp}}" file global_adapt.out screen no

group type1 type 1

compute myRDF all rdf 50 
fix rdf_out all ave/time 24000 1 24000 c_myRDF[*] file rdf_adapt.out mode vector

compute msd_all all msd

variable msd_all_val equal c_msd_all[4]

fix msd_out all ave/time 1000 1 1000 v_msd_all_val file msd_adapt.out mode scalar

# ---- Additional Diagnostics ----


# ---- Additional Diagnostics ----

# (c) Hexatic order parameter:
# Uses hexorder/atom to compute the 2D hexatic order (ideal for probing in-plane hexagonal order).
#compute myHexatic all hexorder/atom degree 6 nnn 6 cutoff 1.2
#fix hexatic_out all ave/time 24000 1 24000 c_myHexatic[*] file hexatic.out mode vector

# (d) 3D Orientational Order:
# This compute captures full three-dimensional bond–orientational order via spherical harmonics.
# Default settings compute several orders (e.g., degrees 5, 4, 6, 8, 10, 12 by default).
compute myOrient all orientorder/atom
compute myOrient1 all reduce ave c_myOrient[1]
compute myOrient2 all reduce ave c_myOrient[2]
compute myOrient3 all reduce ave c_myOrient[3]
compute myOrient4 all reduce ave c_myOrient[4]
compute myOrient5 all reduce ave c_myOrient[5]
fix orient_out all ave/time 24000 1 24000 c_myOrient1 c_myOrient2 c_myOrient3 c_myOrient4 c_myOrient5 file orientorder.out mode scalar

thermo 500
thermo_style custom step temp ke pe etotal

timestep 0.01

velocity all create 0.8 12345
fix flangevin all langevin 0.8 0.8 1.0 12345
fix nve all nve                                 

run 25000

write_data system_after_nvt_adapt.data
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation by assembling configuration, minimization, and NVT scripts.")
    parser.add_argument("--w_c_default", type=float, required=True)
    parser.add_argument("folder", nargs="?", default=None,
                        help="Name of the folder to store this simulation's files (positional).")
    return parser.parse_args()

def create_simulation_folder(folder_name=None):
    if folder_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        folder_name = f"sim_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def write_configuration_file(folder, args):
    config_path = os.path.join(folder, "configuration.txt")
    with open(config_path, "w") as f:
        f.write("Simulation Configuration:\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    return config_path

def write_script(folder, filename, content):
    path = os.path.join(folder, filename)
    with open(path, "w") as f:
        f.write(content)
    return path

def run_assembly_pipeline(folder, args):
    cmd = [
        "python", "assembly_pipeline.py"
    ]
    print("Running assembly_pipeline.py with command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    if os.path.exists("bilayer.data"):
        shutil.move("bilayer.data", os.path.join(folder, "bilayer.data"))
    else:
        raise FileNotFoundError("bilayer.data was not created by assembly_pipeline.py.")

def main():
    args = parse_arguments()
    # Use the provided folder name directly (e.g., "config_1", "config_2", etc.)
    sim_folder = create_simulation_folder(args.folder)
    print(f"Created simulation folder: {sim_folder}")
    write_configuration_file(sim_folder, args)
    run_assembly_pipeline(sim_folder, args)
    
    # Compute the cutoff terms for the pair_coeff lines.
    w_c_default_term = 1.1225 + args.w_c_default      # for tail-tail, unsat-unsat, sat-sat, etc.
    
    # Fill in the minimization and NVT scripts using the provided parameters.
    min_script_filled = MINIMIZATION_SCRIPT.format(
        w_c_default=args.w_c_default,
        w_c_default_term=w_c_default_term
    )
    nvt_script_filled = NVT_SCRIPT.format(
        w_c_default=args.w_c_default,
        w_c_default_term=w_c_default_term,
    )
    
    # Write the filled scripts to the simulation folder.
    min_script_path = write_script(sim_folder, "min.in", min_script_filled)
    print(f"Wrote minimization script to {min_script_path}")
    
    nvt_script_path = write_script(sim_folder, "nvt.in", nvt_script_filled)
    print(f"Wrote NVT script to {nvt_script_path}")
    
    print("Simulation assembly complete. You can now launch LAMMPS using the generated scripts in the folder.")

if __name__ == "__main__":
    main()
