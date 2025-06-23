#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
import networkx as nx
from scipy.spatial import cKDTree

# --- Function to parse the RDF file into blocks ---
def parse_rdf_file(filename):
    """
    Parse the rdf_short.out file into blocks.
    Each block starts with a header line: "TimeStep Number-of-rows"
    followed by that many rows of data. Each data row has 10 numbers:
      [index, radius, g(r1), c(1), g(r2), c(2), g(r3), c(3), g(r4), c(4)]
    Returns a dict mapping timestep -> pandas DataFrame.
    """
    blocks = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue

        tokens = line.split()
        if len(tokens) == 2:
            timestep = float(tokens[0])
            nrows = int(tokens[1])
            i += 1
            block_data = []
            for _ in range(nrows):
                if i >= len(lines):
                    break
                row_line = lines[i].strip()
                if row_line:
                    row_data = [float(tok) for tok in row_line.split()]
                    block_data.append(row_data)
                i += 1
            col_names = [
                "index", "radius",
                "g(r1)", "c(1)",
                "g(r2)", "c(2)",
                "g(r3)", "c(3)",
                "g(r4)", "c(4)"
            ]
            df_block = pd.DataFrame(block_data, columns=col_names)
            blocks[timestep] = df_block
        else:
            i += 1

    return blocks

# --- RDF feature extraction ---
def get_first_rdf_peak_height(rdf_df, r_min=0.1, r_max=3.0):
    subset = rdf_df[(rdf_df["radius"] >= r_min) & (rdf_df["radius"] <= r_max)]
    if subset.empty:
        return np.nan
    return subset["g(r4)"].max()

def get_coord(rdf_df, r_min=0.1, r_max=3.0, target_r=1.5):
    subset = rdf_df[(rdf_df["radius"] >= r_min) & (rdf_df["radius"] <= r_max)]
    if subset.empty:
        return np.nan, np.nan
    r_array = subset["radius"].values
    cn_array = subset["c(4)"].values
    idx_closest = np.abs(r_array - target_r).argmin()
    cn_at_r = cn_array[idx_closest]
    slope = np.polyfit(r_array, cn_array, 1)[0]
    return cn_at_r, slope

# --- Functions for hexatic order parameter calculation ---
def group_lipids(u):
    G = nx.Graph()
    for atom in u.atoms:
        G.add_node(atom.index)
    if hasattr(u, 'bonds'):
        for bond in u.bonds:
            i = bond.atoms[0].index
            j = bond.atoms[1].index
            G.add_edge(i, j)
    else:
        raise RuntimeError("No bond information found in Universe.")
    comps = [comp for comp in nx.connected_components(G) if len(comp) >= 3]
    return comps

def compute_com_for_lipid(u, indices):
    positions = np.array([u.atoms[i].position for i in indices])
    return positions.mean(axis=0)

def local_tangent_basis(r):
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-8:
        raise ValueError("Zero radial vector encountered!")
    r_unit = r / r_norm
    arbitrary = np.array([1, 0, 0]) if abs(r_unit[0]) < 0.9 else np.array([0, 1, 0])
    e1 = arbitrary - np.dot(arbitrary, r_unit) * r_unit
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(r_unit, e1)
    return e1, e2

def compute_hexatic_order_lipids(datafile, trajfile, cutoff=6.0, frame=-1):
    u = mda.Universe(datafile, trajfile, format="LAMMPSDUMP")
    u.trajectory[frame]

    comps = group_lipids(u)
    if not comps:
        return np.nan

    COMs = np.array([compute_com_for_lipid(u, comp) for comp in comps])
    center = COMs.mean(axis=0)
    tree = cKDTree(COMs)
    psi6_vals = []

    for i, pos in enumerate(COMs):
        r_vec = pos - center
        try:
            e1, e2 = local_tangent_basis(r_vec)
        except ValueError:
            continue

        dists, idxs = tree.query(pos, k=7)
        angles = []
        for dist, j in zip(dists[1:], idxs[1:]):
            if dist > cutoff:
                continue
            disp = COMs[j] - pos
            r_unit = r_vec / np.linalg.norm(r_vec)
            proj = disp - np.dot(disp, r_unit) * r_unit
            x_proj = np.dot(proj, e1)
            y_proj = np.dot(proj, e2)
            angles.append(np.arctan2(y_proj, x_proj))

        if angles:
            op = np.mean(np.exp(1j * 6 * np.array(angles)))
            psi6_vals.append(np.abs(op))
        else:
            psi6_vals.append(0.0)

    return np.mean(psi6_vals) if psi6_vals else np.nan

def batch_hexatic(config_root,
                  data_file="system_after_min.data",
                  traj_file="full_traj_short.lammpstrj",
                  cutoff=6.0,
                  frame=-1):
    results = []
    for item in sorted(os.listdir(config_root)):
        folder = os.path.join(config_root, item)
        if not item.startswith("config_") or not os.path.isdir(folder):
            continue

        dpf = os.path.join(folder, data_file)
        tpf = os.path.join(folder, traj_file)
        if os.path.isfile(dpf) and os.path.isfile(tpf):
            try:
                psi6 = compute_hexatic_order_lipids(dpf, tpf, cutoff=cutoff, frame=frame)
            except Exception:
                psi6 = np.nan
        else:
            psi6 = np.nan

        results.append((item, psi6))

    return pd.DataFrame(results, columns=["config", "psi6"])

def main():
    base_dir = "."
    # find and sort config_ folders by the integer after underscore
    configs = sorted(
        [d for d in os.listdir(base_dir)
         if d.startswith("config_") and os.path.isdir(os.path.join(base_dir, d))],
        key=lambda x: int(x.split("_", 1)[1])
    )

    msd_dict = {}
    rdf_blocks = {}

    for cfg in configs:
        folder = os.path.join(base_dir, cfg)

        # 1) MSD: read msd_short.out
        msd_file = os.path.join(folder, "msd_short.out")
        if os.path.isfile(msd_file):
            df = pd.read_csv(
                msd_file,
                delim_whitespace=True,
                names=["TimeStep", "v_msd_all", "v_msd_1", "v_msd_8", "v_msd_9"],
                comment="#"
            )
            msd_dict[cfg] = np.polyfit(df["TimeStep"], df["v_msd_all"], 1)[0]
        else:
            msd_dict[cfg] = np.nan

        # 2) RDF: read rdf_short.out
        rdf_file = os.path.join(folder, "rdf_short.out")
        if os.path.isfile(rdf_file):
            blocks = parse_rdf_file(rdf_file)
            rdf_blocks[cfg] = blocks[max(blocks)] if blocks else None
        else:
            rdf_blocks[cfg] = None

    # Build the main DataFrame
    df = pd.DataFrame(index=configs)
    df["msd"] = pd.Series(msd_dict)
    df["rdf_peak"] = [
        get_first_rdf_peak_height(rdf_blocks[c]) if rdf_blocks[c] is not None else np.nan
        for c in configs
    ]

    # Hexatic ψ₆ from short runs
    df_hex = batch_hexatic(base_dir)
    df = df.join(df_hex.set_index("config"), how="left")

    # Additional RDF‐based features
    cn_vals = []
    slope_vals = []
    for c in configs:
        blk = rdf_blocks[c]
        if blk is not None:
            cn, sl = get_coord(blk)
        else:
            cn, sl = np.nan, np.nan
        cn_vals.append(cn)
        slope_vals.append(sl)

    df["cn_1.5"] = cn_vals
    df["cn_slope"] = slope_vals

    # Ratio features
    df["rdf/msd"] = df["rdf_peak"] / df["msd"]
    df["psi/msd"] = df["psi6"] / df["msd"]
    df["cn1.5/msd"] = df["cn_1.5"] / df["msd"]
    df["cn_slope/msd"] = df["cn_slope"] / df["msd"]

    # Finalize and write out
    df.reset_index(inplace=True)
    df.rename(columns={"index": "config"}, inplace=True)
    output_csv = "features_short_nvt.csv"
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

if __name__ == "__main__":
    main()
