#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from icosphere import icosphere
from sklearn.neighbors import BallTree
from scipy.spatial.transform import Rotation
import sys
import argparse
from collections import Counter

###############################################################################
# ===================== PARSE COMMAND-LINE ARGUMENTS ==========================
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assembly pipeline for spherical bilayer simulation."
    )
    parser.add_argument("--outer_typeOne", type=float, default=0.4,
                        help="Fraction of outer leaflet lipids that are typeOne.")
    parser.add_argument("--outer_typeTwo", type=float, default=0.4,
                        help="Fraction of outer leaflet lipids that are typeTwo.")
    parser.add_argument("--inner_typeThr", type=float, default=0.4,
                        help="Fraction of inner leaflet lipids that are typeThr.")
    parser.add_argument("--inner_typeFour", type=float, default=0.4,
                        help="Fraction of inner leaflet lipids that are typeFour.")
    parser.add_argument("--n_lipids", type=int, default=10000,
                        help="Total number of lipids in the system.")
    return parser.parse_args()

args = parse_args()

###############################################################################
# ======================= PARAMETER BLOCK =====================================
###############################################################################
# TOTAL lipids (can be overridden from the command line)
N_LIPIDS = args.n_lipids

# radial distance difference between outer & inner
THICKNESS = 7

# spacing among head beads on each leaflet
TARGET_DIST = 1.3

# Set up the fractions based on command-line arguments.
# For each leaflet, we assume the sum of the two provided fractions must be ≤ 1.0,
# and the cholesterol fraction (typeFive) is defined as the remainder.
if args.outer_typeOne + args.outer_typeTwo > 1.0:
    sys.exit("Error: outer_typeOne + outer_typeTwo must be ≤ 1.0")
if args.inner_typeThr + args.inner_typeFour > 1.0:
    sys.exit("Error: inner_typeThr + inner_typeFour must be ≤ 1.0")

OUTER_FRACTIONS = {
    "typeOne": args.outer_typeOne,
    "typeTwo": args.outer_typeTwo,
    "typeFive": 1.0 - (args.outer_typeOne + args.outer_typeTwo)
}

INNER_FRACTIONS = {
    "typeThr": args.inner_typeThr,
    "typeFour": args.inner_typeFour,
    "typeFive": 1.0 - (args.inner_typeThr + args.inner_typeFour)
}

###############################################################################
# Species => per-bead atom types (already +1 from your original)
###############################################################################
SPECIES_BEADS = {
    "typeOne":  {"Head":1, "Mid1":3, "Mid2":3, "Tail":2},
    "typeTwo":  {"Head":8, "Mid1":4, "Mid2":4, "Tail":2},
    "typeThr":  {"Head":1, "Mid1":5, "Mid2":5, "Tail":2},
    "typeFour": {"Head":8, "Mid1":6, "Mid2":6, "Tail":2},
    "typeFive": {"Head":9, "Mid1":7, "Mid2":7, "Tail":10},
}

BEAD_RADII = {
    1: 0.95,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 0.80,
    8: 0.95,
    9: 0.76,
    10: 0.80
}

###############################################################################
# BOND TYPES:
#   1 = adjacency (FENE-like) for all species
#   2 = bridging for typeOne & typeThr
#   3 = bridging for typeTwo & typeFour
#   4 = bridging for typeFive
###############################################################################
BOND_TYPE_ADJACENT = 1

def bridging_bond_type(species):
    if species in ("typeOne", "typeThr"):
        return 2
    elif species in ("typeTwo", "typeFour"):
        return 3
    elif species == "typeFive":
        return 4
    else:
        raise ValueError(f"Unknown bridging type for species {species}")

###############################################################################
# MASS for each atom type:
###############################################################################
MASS_DICT = {
    1: 0.95,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 0.80,
    8: 0.95,
    9: 0.76,
    10: 0.80
}

###############################################################################
# Minimal Lipid class
###############################################################################
class Lipid:
    """
    For each lipid, store:
      bead_positions: dict of { "Head":(x,y,z), "Mid1":..., "Mid2":..., "Tail":... }
      species: one of ("typeOne","typeTwo","typeThr","typeFour","typeFive")
    """
    def __init__(self, bead_positions, species):
        self.bead_positions = bead_positions
        self.species = species

    def getPos(self, beadName="CoM"):
        if beadName == "CoM":
            arr = np.array(list(self.bead_positions.values()))
            return arr.mean(axis=0)
        else:
            return np.copy(self.bead_positions[beadName])

###############################################################################
# Build local geometry
###############################################################################
def build_lipid_geometry_local_z(species):
    bead_head = get_atom_type(species, "Head")
    bead_mid1 = get_atom_type(species, "Mid1")
    bead_mid2 = get_atom_type(species, "Mid2")
    bead_tail = get_atom_type(species, "Tail")
    d1 = (BEAD_RADII[bead_head] + BEAD_RADII[bead_mid1]) / 2
    d2 = (BEAD_RADII[bead_mid1] + BEAD_RADII[bead_mid2]) / 2
    d3 = (BEAD_RADII[bead_mid2] + BEAD_RADII[bead_tail]) / 2
    return {
        "Head": np.array([0, 0, 0]),
        "Mid1": np.array([0, 0, d1]),
        "Mid2": np.array([0, 0, d1 + d2]),
        "Tail": np.array([0, 0, d1 + d2 + d3])
    }

###############################################################################
# Create lipid with radial orientation
###############################################################################
def create_lipid_radial(head_pos, species, isOuter, center=np.zeros(3)):
    local = build_lipid_geometry_local_z(species)
    n = head_pos - center
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        n = np.array([0, 0, 1])
        norm_n = 1.0
    else:
        n /= norm_n
    final_dir = -n if isOuter else n
    z_local = np.array([0, 0, 1])
    cross_v = np.cross(z_local, final_dir)
    dot_v = np.dot(z_local, final_dir)
    angle = np.arccos(np.clip(dot_v, -1.0, 1.0))
    if np.linalg.norm(cross_v) > 1e-12:
        cross_v /= np.linalg.norm(cross_v)
        R = Rotation.from_rotvec(angle * cross_v)
        arr = np.array(list(local.values()))
        com = arr.mean(axis=0)
        newpos = {}
        for k, pos in local.items():
            newpos[k] = R.apply(pos - com) + com
        local = newpos
    disp = head_pos - local["Head"]
    for k in local:
        local[k] += disp
    return Lipid(local, species)

###############################################################################
# Build spherical positions for heads (leaflet)
###############################################################################
def build_leaflet_positions(n_lipids, radius, target_dist):
    if n_lipids < 1:
        return np.zeros((0, 3))
    from math import sqrt
    nu = int(round(np.sqrt(n_lipids / 40.0)))
    verts, faces = icosphere(nu)
    F_tot = len(faces)
    desired_faces = max(1, n_lipids // 2)
    if desired_faces > F_tot:
        desired_faces = F_tot
    idx = np.arange(F_tot)
    np.random.shuffle(idx)
    idx = idx[:desired_faces]
    tri = verts[faces[idx]]
    cm = (tri[:, 0] + tri[:, 1] + tri[:, 2]) / 3.0
    if len(cm) > 1:
        tree = BallTree(cm)
        dist_neigh, _ = tree.query(cm, k=6)
        mean_dist = np.mean(dist_neigh[:, 1:])
        scale_factor = target_dist / mean_dist
        cm *= scale_factor
    rcur = np.linalg.norm(cm, axis=1)
    avg_r = rcur.mean() if len(rcur) > 0 else 1.0
    if avg_r > 1e-12:
        cm *= (radius / avg_r)
    npos = len(cm)
    if npos < n_lipids:
        needed = n_lipids - npos
        extra = []
        while len(extra) < needed:
            i2 = np.random.randint(0, npos)
            jitter = np.random.normal(scale=0.01, size=3)
            extra.append(cm[i2] + jitter)
        cm = np.vstack([cm, extra])
    elif npos > n_lipids:
        cm = cm[:n_lipids]
    return cm

###############################################################################
# Build species list for each leaflet
###############################################################################
def assign_leaflet_species(n_lipids, fractions_dict):
    sp_keys = list(fractions_dict.keys())
    assigned = 0
    counts = {}
    for sp in sp_keys:
        csp = int(round(fractions_dict[sp] * n_lipids))
        counts[sp] = csp
        assigned += csp
    if assigned < n_lipids:
        leftover = n_lipids - assigned
        first_sp = sp_keys[0]
        counts[first_sp] += leftover
    elif assigned > n_lipids:
        excess = assigned - n_lipids
        first_sp = sp_keys[0]
        counts[first_sp] = max(0, counts[first_sp] - excess)
    species_list = []
    for sp in sp_keys:
        species_list += [sp] * counts[sp]
    np.random.shuffle(species_list)
    return species_list

###############################################################################
# Mapping from species+beadName to an integer atom type
###############################################################################
def get_atom_type(species, beadName):
    return SPECIES_BEADS[species][beadName]

###############################################################################
# Build adjacency + bridging bonds for a single lipid
###############################################################################
def build_bonds_for_lipid(bead_ids, species):
    H = bead_ids["Head"]
    M1 = bead_ids["Mid1"]
    M2 = bead_ids["Mid2"]
    T = bead_ids["Tail"]
    btype_bridge = bridging_bond_type(species)
    bonds = []
    bonds.append((BOND_TYPE_ADJACENT, H, M1))
    bonds.append((BOND_TYPE_ADJACENT, M1, M2))
    bonds.append((BOND_TYPE_ADJACENT, M2, T))
    bonds.append((btype_bridge, H, M2))
    bonds.append((btype_bridge, M1, T))
    return bonds

###############################################################################
# Write LAMMPS data
###############################################################################
def write_lammps_data(fname, lipids):
    atom_id = 1
    bond_id = 1
    mol_id = 1
    all_atoms = []
    all_bonds = []
    for lip in lipids:
        beadNames = ["Head", "Mid1", "Mid2", "Tail"]
        bead_ids = {}
        for bn in beadNames:
            atype = get_atom_type(lip.species, bn)
            x, y, z = lip.getPos(bn)
            charge = 0.0
            all_atoms.append((atom_id, mol_id, atype, charge, x, y, z))
            bead_ids[bn] = atom_id
            atom_id += 1
        lbonds = build_bonds_for_lipid(bead_ids, lip.species)
        for (btype, aA, aB) in lbonds:
            all_bonds.append((bond_id, btype, aA, aB))
            bond_id += 1
        mol_id += 1
    N_atoms = len(all_atoms)
    N_bonds = len(all_bonds)
    N_angles = 0
    N_dihedrals = 0
    N_impropers = 0
    used_types = set([a[2] for a in all_atoms])
    max_type = max(used_types) if used_types else 1
    coords = np.array([(a[4], a[5], a[6]) for a in all_atoms])
    xyz_min = coords.min(axis=0) - 20.0
    xyz_max = coords.max(axis=0) + 20.0
    mass_and_comment = {
        1:  (1.0, "# unsat_head"),
        2:  (1.0, "# tail"),
        3:  (1.0, "# unsat_outer_mid"),
        4:  (1.0, "# sat_outer_mid"),
        5:  (1.0, "# unsat_inner_mid"),
        6:  (1.0, "# sat_inner_mid"),
        7:  (0.8, "# chol_mid"),
        8:  (1.0, "# sat_head"),
        9:  (0.8, "# chol_head"),
        10: (0.8, "# chol_tail")
    }
    with open(fname, "w") as f:
        f.write("LAMMPS data file: Spherical Bilayer with 5 new species (shifted +1)\n\n")
        f.write(f"{N_atoms} atoms\n")
        f.write(f"{N_bonds} bonds\n")
        f.write(f"{N_angles} angles\n")
        f.write(f"{N_dihedrals} dihedrals\n")
        f.write(f"{N_impropers} impropers\n\n")
        f.write(f"{max_type} atom types\n")
        f.write("4 bond types\n")
        f.write("0 angle types\n0 dihedral types\n0 improper types\n\n")
        f.write(f"{xyz_min[0]:.3f} {xyz_max[0]:.3f} xlo xhi\n")
        f.write(f"{xyz_min[1]:.3f} {xyz_max[1]:.3f} ylo yhi\n")
        f.write(f"{xyz_min[2]:.3f} {xyz_max[2]:.3f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for itype in range(1, 11):
            mass_value, comment = mass_and_comment[itype]
            f.write(f"{itype} {mass_value:.4f} {comment}\n")
        f.write("\nAtoms # full\n\n")
        for (aid, mid, atype, charge, xx, yy, zz) in all_atoms:
            f.write(f"{aid} {mid} {atype} {charge:.3f} {xx:.5f} {yy:.5f} {zz:.5f}\n")
        f.write("\nBonds\n\n")
        for (bid, btype, aA, aB) in all_bonds:
            f.write(f"{bid} {btype} {aA} {aB}\n")
    print(f"Wrote LAMMPS data => {fname}")
    print(f"  #atoms = {N_atoms}, #bonds = {N_bonds}")

###############################################################################
# Main
###############################################################################
def main():
    # If a command-line argument is provided, use it as N_LIPIDS; otherwise, default.
    if len(sys.argv) > 1:
        try:
            nTotal = int(sys.argv[1])
        except ValueError:
            nTotal = N_LIPIDS
    else:
        nTotal = N_LIPIDS

    # define outer & inner radius
    R_outer = nTotal / 500.0
    R_inner = R_outer - THICKNESS
    if R_inner <= 0:
        raise ValueError("Negative R_inner => reduce THICKNESS or increase N_LIPIDS.")

    # Determine fraction using ratio of areas:
    frac_outer = R_outer**2 / (R_outer**2 + R_inner**2)
    N_outer = int(round(frac_outer * nTotal))
    N_inner = nTotal - N_outer

    # Build HEAD positions
    outer_heads = build_leaflet_positions(N_outer, R_outer, TARGET_DIST)
    inner_heads = build_leaflet_positions(N_inner, R_inner, TARGET_DIST)

    # Assign species using the fractions provided via command-line
    outer_species_list = assign_leaflet_species(N_outer, OUTER_FRACTIONS)
    inner_species_list = assign_leaflet_species(N_inner, INNER_FRACTIONS)

    # Build Lipid objects
    ulipids = []
    for i, sp in enumerate(outer_species_list):
        lip = create_lipid_radial(outer_heads[i], sp, True, center=np.zeros(3))
        ulipids.append(lip)
    dlipids = []
    for i, sp in enumerate(inner_species_list):
        lip = create_lipid_radial(inner_heads[i], sp, False, center=np.zeros(3))
        dlipids.append(lip)
    all_lipids = ulipids + dlipids
    print(f"Created total {len(all_lipids)} lipids => outer={len(ulipids)}, inner={len(dlipids)}")
    print("Outer species =>", Counter([l.species for l in ulipids]))
    print("Inner species =>", Counter([l.species for l in dlipids]))
    # Write LAMMPS data file
    OUTPUT_FILE = "bilayer.data"
    write_lammps_data(OUTPUT_FILE, all_lipids)
    # Optional: Quick 3D check
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    oh = np.array([l.getPos("Head") for l in ulipids])
    ih = np.array([l.getPos("Head") for l in dlipids])
    ax.scatter(oh[:, 0], oh[:, 1], oh[:, 2], s=5, label='Outer leaflet')
    ax.scatter(ih[:, 0], ih[:, 1], ih[:, 2], s=5, label='Inner leaflet')
    ax.legend()
    ax.set_title("Spherical Bilayer, 5 new species (area-based leaflet sizes)")
    plt.close()
    #plt.show()

if __name__ == "__main__":
    main()
