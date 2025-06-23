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
bond_coeff 2 harmonic  {k_bend_unsat} 4.0
# bridging for sat - HALF K-VALUE FROM LITERATURE
bond_coeff 3 harmonic  {k_bend_sat} 4.0
# bridging for chol - HALF K-VALUE FROM LITERATURE
bond_coeff 4 harmonic  {k_bend_chol} 4.0

pair_style cosine/squared 3.0

# ----------------------------------------------------------------
# (A) Head–anything => purely repulsive WCA
#     Heads = types 1,8,9
# ----------------------------------------------------------------
pair_coeff 1 1 1.0  1.0663  1.0663 wca
pair_coeff 1 2 1.0  1.0934  1.0934 wca
pair_coeff 1 3 1.0  1.0934  1.0934 wca
pair_coeff 1 4 1.0  1.0934  1.0934 wca
pair_coeff 1 5 1.0  1.0934  1.0934 wca
pair_coeff 1 6 1.0  1.0934  1.0934 wca
pair_coeff 1 7 1.0  0.9807  0.9807 wca
pair_coeff 1 8 1.0  1.0663  1.0663 wca
pair_coeff 1 9 1.0  0.9592  0.9592 wca
pair_coeff 1 10 1.0  0.9807  0.9807 wca

# 2 with 8..9
pair_coeff 2 8 1.0  1.0934  1.0934 wca
pair_coeff 2 9 1.0  0.9863  0.9863 wca

# 3 with 8..9
pair_coeff 3 8 1.0  1.0934  1.0934 wca
pair_coeff 3 9 1.0  0.9863  0.9863 wca

# 4 with 8..9
pair_coeff 4 8 1.0  1.0934  1.0934 wca
pair_coeff 4 9 1.0  0.9863  0.9863 wca

# 5 with 8..9
pair_coeff 5 8 1.0  1.0934  1.0934 wca
pair_coeff 5 9 1.0  0.9863  0.9863 wca

# 6 with 8..9
pair_coeff 6 8 1.0  1.0934  1.0934 wca
pair_coeff 6 9 1.0  0.9863  0.9863 wca

# 7 with 8..9
pair_coeff 7 8 1.0  0.9807  0.9807 wca
pair_coeff 7 9 1.0  0.8739  0.8739 wca

# 8 with 8..10
pair_coeff 8 8 1.0  1.0663  1.0663 wca
pair_coeff 8 9 1.0  0.9592  0.9592 wca
pair_coeff 8 10 1.0  0.9807  0.9807 wca

# 9 with 9..10
pair_coeff 9 9 1.0  0.8531  0.8531 wca
pair_coeff 9 10 1.0  0.8739  0.8739 wca

# ----------------------------------------------------------------
# (B) Cross-leaflet mid–mid => purely repulsive WCA
# ----------------------------------------------------------------
pair_coeff 3 5 1.0  1.1225  1.1225 wca
pair_coeff 3 6 1.0  1.1225  1.1225 wca
pair_coeff 4 5 1.0  1.1225  1.1225 wca
pair_coeff 4 6 1.0  1.1225  1.1225 wca

# ----------------------------------------------------------------
# (C) Everything else => cos² with w_c from the old table
# ----------------------------------------------------------------
# (C1) tail-tail => (2,2), 
pair_coeff 2 2 1.0  1.1225  {w_c_default_term} wca

# (C2) unsat–unsat => 
pair_coeff 2 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 3 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 5 1.0  1.1225  {w_c_default_term} wca
pair_coeff 5 5 1.0  1.1225  {w_c_default_term} wca

# (C3) sat–sat => 
pair_coeff 2 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 4 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 6 1.0  1.1225  {w_c_default_term} wca
pair_coeff 6 6 1.0  1.1225  {w_c_default_term} wca

# (C4) tail–chol => (sigma average=1.010)
pair_coeff 2 7  1.0  1.010  {w_c_default_term2} wca
pair_coeff 2 10 1.0  1.010  {w_c_default_term2} wca

# (C5) chol–chol => (sigma average=0.898)
pair_coeff 7 7  1.0  0.898  {w_c_default_term3} wca
pair_coeff 7 10 1.0  0.898  {w_c_default_term3} wca
pair_coeff 10 10 1.0  0.898  {w_c_default_term3} wca

# (C6) unsat–sat => 
pair_coeff 3 4 1.0  1.1225  {w_c_U_S_term} wca
pair_coeff 5 6 1.0  1.1225  {w_c_U_S_term} wca

# (C7) unsat–chol => 
pair_coeff 3 7  1.0  1.010  {w_c_U_C_term} wca
pair_coeff 3 10 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 7  1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 10 1.0  1.010  {w_c_U_C_term} wca

# (C8) sat–chol => 
pair_coeff 4 7  1.0  1.010  {w_c_C_S_term} wca
pair_coeff 4 10 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 7  1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 10 1.0  1.010  {w_c_C_S_term} wca

comm_modify cutoff 7.0
dump            1 all custom 50 traj_min.lammpstrj id mol type x y z ix iy iz
thermo_style    custom step pe etotal vol epair ebond eangle
thermo          40

minimize 1.0e-7 1.0e-9 10000 300000

write_data system_after_min.data
"""

# ----- Updated NVT SCRIPT with placeholders -----
NVT_SCRIPT_SHORT = r"""# ----------------- Init Section -----------------
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
bond_coeff 2 harmonic  {k_bend_unsat} 4.0
bond_coeff 3 harmonic  {k_bend_sat} 4.0
bond_coeff 4 harmonic  {k_bend_chol} 4.0

pair_style cosine/squared 3.0

# (A) Head–anything => purely repulsive WCA
pair_coeff 1 1 1.0  1.0663  1.0663 wca
pair_coeff 1 2 1.0  1.0934  1.0934 wca
pair_coeff 1 3 1.0  1.0934  1.0934 wca
pair_coeff 1 4 1.0  1.0934  1.0934 wca
pair_coeff 1 5 1.0  1.0934  1.0934 wca
pair_coeff 1 6 1.0  1.0934  1.0934 wca
pair_coeff 1 7 1.0  0.9807  0.9807 wca
pair_coeff 1 8 1.0  1.0663  1.0663 wca
pair_coeff 1 9 1.0  0.9592  0.9592 wca
pair_coeff 1 10 1.0  0.9807  0.9807 wca

pair_coeff 2 8 1.0  1.0934  1.0934 wca
pair_coeff 2 9 1.0  0.9863  0.9863 wca

pair_coeff 3 8 1.0  1.0934  1.0934 wca
pair_coeff 3 9 1.0  0.9863  0.9863 wca

pair_coeff 4 8 1.0  1.0934  1.0934 wca
pair_coeff 4 9 1.0  0.9863  0.9863 wca

pair_coeff 5 8 1.0  1.0934  1.0934 wca
pair_coeff 5 9 1.0  0.9863  0.9863 wca

pair_coeff 6 8 1.0  1.0934  1.0934 wca
pair_coeff 6 9 1.0  0.9863  0.9863 wca

pair_coeff 7 8 1.0  0.9807  0.9807 wca
pair_coeff 7 9 1.0  0.8739  0.8739 wca

pair_coeff 8 8 1.0  1.0663  1.0663 wca
pair_coeff 8 9 1.0  0.9592  0.9592 wca
pair_coeff 8 10 1.0  0.9807  0.9807 wca

pair_coeff 9 9 1.0  0.8531  0.8531 wca
pair_coeff 9 10 1.0  0.8739  0.8739 wca

# (B) Cross-leaflet mid–mid => purely repulsive WCA
pair_coeff 3 5 1.0  1.1225  1.1225 wca
pair_coeff 3 6 1.0  1.1225  1.1225 wca
pair_coeff 4 5 1.0  1.1225  1.1225 wca
pair_coeff 4 6 1.0  1.1225  1.1225 wca

# (C) Everything else => cos² with w_c from the old table
# (C1) tail-tail => (2,2)
pair_coeff 2 2 1.0  1.1225  {w_c_default_term} wca

# (C2) unsat–unsat => 
pair_coeff 2 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 3 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 5 1.0  1.1225  {w_c_default_term} wca
pair_coeff 5 5 1.0  1.1225  {w_c_default_term} wca

# (C3) sat–sat => 
pair_coeff 2 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 4 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 6 1.0  1.1225  {w_c_default_term} wca
pair_coeff 6 6 1.0  1.1225  {w_c_default_term} wca

# (C4) tail–chol => (sigma average=1.010)
pair_coeff 2 7  1.0  1.010  {w_c_default_term2} wca
pair_coeff 2 10 1.0  1.010  {w_c_default_term2} wca

# (C5) chol–chol =>  (sigma average=0.898)
pair_coeff 7 7  1.0  0.898  {w_c_default_term3} wca
pair_coeff 7 10 1.0  0.898  {w_c_default_term3} wca
pair_coeff 10 10 1.0  0.898  {w_c_default_term3} wca

# (C6) unsat–sat => 
pair_coeff 3 4 1.0  1.1225  {w_c_U_S_term} wca
pair_coeff 5 6 1.0  1.1225  {w_c_U_S_term} wca

# (C7) unsat–chol =
pair_coeff 3 7 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 3 10 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 7 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 10 1.0  1.010  {w_c_U_C_term} wca

# (C8) sat–chol => 
pair_coeff 4 7 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 4 10 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 7 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 10 1.0  1.010  {w_c_C_S_term} wca

comm_modify cutoff 7.0
# ---- Normal Coordinate Trajectory Dump ----
# Dump atom id, type, and positions every 1000 steps.
dump           1 all custom 300 full_traj_short.lammpstrj id type x y z
dump_modify    1 sort id


# ---- Define Global Variables to be Dumped ----
variable ke   equal ke         # kinetic energy
variable pe   equal pe         # potential energy
variable tot  equal etotal     # total energy
variable temp equal temp       # temperature

# Define a custom variable for the current simulation step
variable simstep equal step

fix global_print all print 1000 "Step: ${{simstep}}  KE: ${{ke}}  PE: ${{pe}}  Tot: ${{tot}}  Temp: ${{temp}}" file global_short.out screen no

group type1 type 1
group type8 type 8
group type9 type 9

compute myRDF all rdf 50 1 8 1 9 8 9 * *
fix rdf_out all ave/time 1000 1 1000 c_myRDF[*] file rdf_short.out mode vector

compute msd_all all msd
compute msd_1 type1 msd
compute msd_8 type8 msd
compute msd_9 type9 msd

variable msd_all_val equal c_msd_all[4]
variable msd_1_val   equal c_msd_1[4]
variable msd_8_val   equal c_msd_8[4]
variable msd_9_val   equal c_msd_9[4]

fix msd_out all ave/time 1000 1 1000 v_msd_all_val v_msd_1_val v_msd_8_val v_msd_9_val file msd_short.out mode scalar

thermo 500
thermo_style custom step temp ke pe etotal

timestep 0.01

velocity all create {Temperature} 12345
fix flangevin all langevin {Temperature} {Temperature} 1.0 12346
fix nve all nve                                 

run 25000

write_data system_after_short_nvt.data
"""

# ----- Updated NVT SCRIPT with placeholders -----
NVT_SCRIPT_LONG = r"""# ----------------- Init Section -----------------
units           lj
dimension       3
atom_style      full

bond_style      hybrid fene harmonic 
angle_style     none

pair_style cosine/squared 3.0

special_bonds   lj 0.0 1.0 1.0

neighbor        0.4 bin
neigh_modify    every 2 delay 0 check yes

read_data system_after_short_nvt.data

bond_style hybrid fene harmonic 
bond_coeff 1 fene     30.0 1.5 1.0 1.0
bond_coeff 2 harmonic  {k_bend_unsat} 4.0
bond_coeff 3 harmonic  {k_bend_sat} 4.0
bond_coeff 4 harmonic  {k_bend_chol} 4.0

pair_style cosine/squared 3.0

# (A) Head–anything => purely repulsive WCA
pair_coeff 1 1 1.0  1.0663  1.0663 wca
pair_coeff 1 2 1.0  1.0934  1.0934 wca
pair_coeff 1 3 1.0  1.0934  1.0934 wca
pair_coeff 1 4 1.0  1.0934  1.0934 wca
pair_coeff 1 5 1.0  1.0934  1.0934 wca
pair_coeff 1 6 1.0  1.0934  1.0934 wca
pair_coeff 1 7 1.0  0.9807  0.9807 wca
pair_coeff 1 8 1.0  1.0663  1.0663 wca
pair_coeff 1 9 1.0  0.9592  0.9592 wca
pair_coeff 1 10 1.0  0.9807  0.9807 wca

pair_coeff 2 8 1.0  1.0934  1.0934 wca
pair_coeff 2 9 1.0  0.9863  0.9863 wca

pair_coeff 3 8 1.0  1.0934  1.0934 wca
pair_coeff 3 9 1.0  0.9863  0.9863 wca

pair_coeff 4 8 1.0  1.0934  1.0934 wca
pair_coeff 4 9 1.0  0.9863  0.9863 wca

pair_coeff 5 8 1.0  1.0934  1.0934 wca
pair_coeff 5 9 1.0  0.9863  0.9863 wca

pair_coeff 6 8 1.0  1.0934  1.0934 wca
pair_coeff 6 9 1.0  0.9863  0.9863 wca

pair_coeff 7 8 1.0  0.9807  0.9807 wca
pair_coeff 7 9 1.0  0.8739  0.8739 wca

pair_coeff 8 8 1.0  1.0663  1.0663 wca
pair_coeff 8 9 1.0  0.9592  0.9592 wca
pair_coeff 8 10 1.0  0.9807  0.9807 wca

pair_coeff 9 9 1.0  0.8531  0.8531 wca
pair_coeff 9 10 1.0  0.8739  0.8739 wca

# (B) Cross-leaflet mid–mid => purely repulsive WCA
pair_coeff 3 5 1.0  1.1225  1.1225 wca
pair_coeff 3 6 1.0  1.1225  1.1225 wca
pair_coeff 4 5 1.0  1.1225  1.1225 wca
pair_coeff 4 6 1.0  1.1225  1.1225 wca

# (C) Everything else => cos² with w_c from the old table
# (C1) tail-tail => (2,2)
pair_coeff 2 2 1.0  1.1225  {w_c_default_term} wca

# (C2) unsat–unsat => 
pair_coeff 2 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 3 3 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 5 1.0  1.1225  {w_c_default_term} wca
pair_coeff 5 5 1.0  1.1225  {w_c_default_term} wca

# (C3) sat–sat => 
pair_coeff 2 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 4 4 1.0  1.1225  {w_c_default_term} wca
pair_coeff 2 6 1.0  1.1225  {w_c_default_term} wca
pair_coeff 6 6 1.0  1.1225  {w_c_default_term} wca

# (C4) tail–chol => (sigma average=1.010)
pair_coeff 2 7  1.0  1.010  {w_c_default_term2} wca
pair_coeff 2 10 1.0  1.010  {w_c_default_term2} wca

# (C5) chol–chol =>  (sigma average=0.898)
pair_coeff 7 7  1.0  0.898  {w_c_default_term3} wca
pair_coeff 7 10 1.0  0.898  {w_c_default_term3} wca
pair_coeff 10 10 1.0  0.898  {w_c_default_term3} wca

# (C6) unsat–sat => 
pair_coeff 3 4 1.0  1.1225  {w_c_U_S_term} wca
pair_coeff 5 6 1.0  1.1225  {w_c_U_S_term} wca

# (C7) unsat–chol =
pair_coeff 3 7 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 3 10 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 7 1.0  1.010  {w_c_U_C_term} wca
pair_coeff 5 10 1.0  1.010  {w_c_U_C_term} wca

# (C8) sat–chol => 
pair_coeff 4 7 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 4 10 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 7 1.0  1.010  {w_c_C_S_term} wca
pair_coeff 6 10 1.0  1.010  {w_c_C_S_term} wca

comm_modify cutoff 7.0
# ---- Normal Coordinate Trajectory Dump ----
# Dump atom id, type, and positions every 1000 steps.
dump           1 all custom 300 full_traj_long.lammpstrj id type x y z
dump_modify    1 sort id


# ---- Define Global Variables to be Dumped ----
variable ke   equal ke         # kinetic energy
variable pe   equal pe         # potential energy
variable tot  equal etotal     # total energy
variable temp equal temp       # temperature

# Define a custom variable for the current simulation step
variable simstep equal step

fix global_print all print 10000 "Step: ${{simstep}}  KE: ${{ke}}  PE: ${{pe}}  Tot: ${{tot}}  Temp: ${{temp}}" file global_long.out screen no

thermo 500
thermo_style custom step temp ke pe etotal

timestep 0.01

velocity all create {Temperature} 12345
fix flangevin all langevin {Temperature} {Temperature} 1.0 12346
fix nve all nve                                 

run 475000

write_data system_after_nvt_long.data
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation by assembling configuration, minimization, and NVT scripts.")
    parser.add_argument("--k_bend_saturated", type=float, required=True)
    parser.add_argument("--k_bend_unsaturated", type=float, required=True)
    parser.add_argument("--k_bend_cholesterol", type=float, required=True)
    parser.add_argument("--w_c_default", type=float, required=True)
    parser.add_argument("--w_c_U_S", type=float, required=True)
    parser.add_argument("--w_c_U_C", type=float, required=True)
    parser.add_argument("--w_c_C_S", type=float, required=True)
    parser.add_argument("--Temperature", type=float, required=True)
    parser.add_argument("--outer_typeOne", type=float, required=True)
    parser.add_argument("--outer_typeTwo", type=float, required=True)
    parser.add_argument("--inner_typeThr", type=float, required=True)
    parser.add_argument("--inner_typeFour", type=float, required=True)
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
        "python", "3_assembly_pipeline.py",
        "--outer_typeOne", f"{args.outer_typeOne}",
        "--outer_typeTwo", f"{args.outer_typeTwo}",
        "--inner_typeThr", f"{args.inner_typeThr}",
        "--inner_typeFour", f"{args.inner_typeFour}"
    ]
    print("Running 3_assembly_pipeline.py with command:")
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
    w_c_default_term2 = 1.010 + args.w_c_default         # for tail-chol (C4)
    w_c_default_term3 = 0.898 + args.w_c_default         # for chol-chol (C5)
    w_c_U_S_term = 1.1225 + args.w_c_U_S                  # for unsat-sat (C6)
    w_c_U_C_term = 1.010 + args.w_c_U_C                  # for unsat-chol (C7)
    w_c_C_S_term = 1.010 + args.w_c_C_S                  # for sat-chol (C8)
    
    # Fill in the minimization and NVT scripts using the provided parameters.
    min_script_filled = MINIMIZATION_SCRIPT.format(
        k_bend_unsat=args.k_bend_unsaturated,
        k_bend_sat=args.k_bend_saturated,
        k_bend_chol=args.k_bend_cholesterol,
        w_c_default=args.w_c_default,
        w_c_default_term=w_c_default_term,
        w_c_default_term2=w_c_default_term2,
        w_c_default_term3=w_c_default_term3,
        w_c_U_S_term=w_c_U_S_term,
        w_c_U_C_term=w_c_U_C_term,
        w_c_C_S_term=w_c_C_S_term
    )
    nvt_script_filled_short = NVT_SCRIPT_SHORT.format(
        k_bend_unsat=args.k_bend_unsaturated,
        k_bend_sat=args.k_bend_saturated,
        k_bend_chol=args.k_bend_cholesterol,
        w_c_default=args.w_c_default,
        w_c_default_term=w_c_default_term,
        w_c_default_term2=w_c_default_term2,
        w_c_default_term3=w_c_default_term3,
        w_c_U_S_term=w_c_U_S_term,
        w_c_U_C_term=w_c_U_C_term,
        w_c_C_S_term=w_c_C_S_term,
        Temperature=args.Temperature
    )
    nvt_script_filled_long = NVT_SCRIPT_LONG.format(
        k_bend_unsat=args.k_bend_unsaturated,
        k_bend_sat=args.k_bend_saturated,
        k_bend_chol=args.k_bend_cholesterol,
        w_c_default=args.w_c_default,
        w_c_default_term=w_c_default_term,
        w_c_default_term2=w_c_default_term2,
        w_c_default_term3=w_c_default_term3,
        w_c_U_S_term=w_c_U_S_term,
        w_c_U_C_term=w_c_U_C_term,
        w_c_C_S_term=w_c_C_S_term,
        Temperature=args.Temperature
    )
    
    # Write the filled scripts to the simulation folder.
    min_script_path = write_script(sim_folder, "min.in", min_script_filled)
    print(f"Wrote minimization script to {min_script_path}")
    
    nvt_script_path_short = write_script(sim_folder, "nvt_short.in", nvt_script_filled_short)
    nvt_script_path_long = write_script(sim_folder, "nvt_long.in", nvt_script_filled_long)
    print(f"Wrote NVT script to {nvt_script_path_short}, {nvt_script_path_long}")
    
    print("Simulation assembly complete. You can now launch LAMMPS using the generated scripts in the folder.")

if __name__ == "__main__":
    main()
