#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
import networkx as nx
from scipy.spatial import cKDTree

# --- Function to parse the RDF file into blocks ---
def parse_rdf_file(filename):
    """
    Parse the rdf.out file into blocks.
    Each block starts with a header line: "TimeStep Number-of-rows"
    followed by that many rows of data. Each data row should have 10 numbers:
      [index, radius, g(r1), c(1), g(r2), c(2), g(r3), c(3), g(r4), c(4)]
    Returns a dictionary mapping timestep (float) -> pandas DataFrame.
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
        # Expect header lines with two tokens
        if len(tokens) == 2:
            timestep = float(tokens[0])
            nrows = int(tokens[1])
            i += 1  # move to data rows
            block_data = []
            for _ in range(nrows):
                if i >= len(lines):
                    break
                row_line = lines[i].strip()
                if row_line:
                    row_data = [float(tok) for tok in row_line.split()]
                    block_data.append(row_data)
                i += 1
            col_names = ["index", "radius", "g(r1)", "c(1)"]
            df_block = pd.DataFrame(block_data, columns=col_names)
            blocks[timestep] = df_block
        else:
            i += 1
    return blocks

# --- RDF feature extraction functions ---
def get_first_rdf_peak_height(rdf_df, r_min=0.1, r_max=3.0):
    """
    Given an RDF DataFrame with columns "radius" and "g(r4)",
    returns the maximum of g(r4) within the interval [r_min, r_max].
    """
    subset = rdf_df[(rdf_df["radius"] >= r_min) & (rdf_df["radius"] <= r_max)]
    if subset.empty:
        return np.nan
    return subset["g(r1)"].max()

def get_coord(rdf_df, r_min=0.1, r_max=3.0, target_r=1.5):
    """
    Given an RDF DataFrame with columns "radius" and "c(4)",
    returns the coordination number at r=target_r (or the nearest value)
    and the overall slope of c(4) vs radius in the interval [r_min, r_max].
    """
    subset = rdf_df[(rdf_df["radius"] >= r_min) & (rdf_df["radius"] <= r_max)]
    if subset.empty:
        return np.nan, np.nan
    r_array = subset["radius"].values
    cn_array = subset["c(1)"].values
    idx_closest = np.abs(r_array - target_r).argmin()
    cn_at_r = cn_array[idx_closest]
    slope = np.polyfit(r_array, cn_array, deg=1)[0]
    return cn_at_r, slope

# --- Functions for hexatic order parameter calculation ---
def group_lipids(u):
    """
    Build a graph from the bonds in the Universe.
    Each connected component of the graph is assumed to be one lipid.
    """
    G = nx.Graph()
    for atom in u.atoms:
        G.add_node(atom.index)
    if hasattr(u, 'bonds'):
        for bond in u.bonds:
            i = bond.atoms[0].index
            j = bond.atoms[1].index
            G.add_edge(i, j)
    else:
        raise RuntimeError("No bond information found in Universe. "
                           "Ensure your LAMMPS data file includes bonds.")
    # Each connected component represents a lipid
    lipid_components = list(nx.connected_components(G))
    lipid_components = [comp for comp in lipid_components if len(comp) >= 3]
    return lipid_components

def compute_com_for_lipid(u, indices):
    """
    Compute the center-of-mass for a lipid defined by a set of atom indices.
    Assumes equal mass for all atoms.
    """
    positions = np.array([u.atoms[i].position for i in indices])
    return positions.mean(axis=0)

def local_tangent_basis(r):
    """
    Given a radial vector r from the sphere center, compute two perpendicular unit
    vectors (e1, e2) spanning the tangent plane at that point.
    """
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
    """
    Load the LAMMPS data and trajectory and compute the hexatic order parameter
    for the lipid COMs.
    """
    u = mda.Universe(datafile, trajfile, format="LAMMPSDUMP")
    u.trajectory[frame]  # Select the desired frame

    lipid_components = group_lipids(u)
    lipid_COMs = [compute_com_for_lipid(u, comp) for comp in lipid_components]
    lipid_COMs = np.array(lipid_COMs)
    center = lipid_COMs.mean(axis=0)
    tree = cKDTree(lipid_COMs)
    psi6_values = []
    n_lipids = len(lipid_COMs)
    
    for i in range(n_lipids):
        pos_i = lipid_COMs[i]
        r_vec = pos_i - center
        try:
            e1, e2 = local_tangent_basis(r_vec)
        except ValueError:
            continue
        dists, indices = tree.query(pos_i, k=7)  # k includes self
        neighbor_angles = []
        for dist, j in zip(dists[1:], indices[1:]):  # Skip self
            if dist > cutoff:
                continue
            disp = lipid_COMs[j] - pos_i
            r_unit = r_vec / np.linalg.norm(r_vec)
            proj = disp - np.dot(disp, r_unit) * r_unit
            x_proj = np.dot(proj, e1)
            y_proj = np.dot(proj, e2)
            angle = np.arctan2(y_proj, x_proj)
            neighbor_angles.append(angle)
        if len(neighbor_angles) == 0:
            psi6_values.append(0.0)
        else:
            op_complex = np.mean(np.exp(1j * 6 * np.array(neighbor_angles)))
            psi6_values.append(np.abs(op_complex))
    
    if len(psi6_values) == 0:
        return np.nan
    return np.mean(psi6_values)

def batch_hexatic(config_root, data_file="bilayer.data", traj_file="full_traj.lammpstrj", cutoff=6.0, frame=-1):
    """
    Loop over config directories (those with names starting with 'config_') in config_root.
    For each, if the required data and trajectory files exist, compute the hexatic order parameter.
    Returns a DataFrame with columns ["config", "psi6"].
    """
    results = []
    for item in sorted(os.listdir(config_root)):
        if not item.startswith("config_"):
            continue
        folder = os.path.join(config_root, item)
        data_path = os.path.join(folder, data_file)
        traj_path = os.path.join(folder, traj_file)
        if not (os.path.isdir(folder) and os.path.isfile(data_path) and os.path.isfile(traj_path)):
            continue
        try:
            psi6_val = compute_hexatic_order_lipids(data_path, traj_path, cutoff=cutoff, frame=frame)
            results.append((item, psi6_val))
        except Exception:
            results.append((item, np.nan))
    return pd.DataFrame(results, columns=["config", "psi6"])

# --- Main processing function ---
def main():
    base_dir = "."  # Adjust if needed
    # Gather config directories (assumed to be named "config_<number>")
    configs = [d for d in os.listdir(base_dir)
               if d.startswith("config_") and os.path.isdir(os.path.join(base_dir, d))]
    configs.sort(key=lambda x: int(x.split('_')[1]))

    msd_full_dict = {}
    rdf_data_dict = {}

    for config in configs:
        config_path = os.path.join(base_dir, config)
        # Process MSD file if it exists
        msd_file = os.path.join(config_path, "msd_adapt.out")
        if os.path.isfile(msd_file):
            try:
                msd_df = pd.read_csv(msd_file, comment='#', delim_whitespace=True, header=None)
                msd_df.columns = ["TimeStep", "v_msd_all_val"]
                slope_full, _ = np.polyfit(msd_df["TimeStep"], msd_df["v_msd_all_val"], 1)
                msd_full_dict[config] = slope_full
            except Exception:
                msd_full_dict[config] = np.nan
        else:
            msd_full_dict[config] = np.nan

        # Process RDF file if it exists
        rdf_file = os.path.join(config_path, "rdf_adapt.out")
        if os.path.isfile(rdf_file):
            rdf_blocks = parse_rdf_file(rdf_file)
            if rdf_blocks:
                latest_ts = max(rdf_blocks.keys())
                rdf_data_dict[config] = rdf_blocks[latest_ts]
            else:
                rdf_data_dict[config] = None
        else:
            rdf_data_dict[config] = None

    # Build the merged dataframe using the config names as the index.
    merged_df = pd.DataFrame(index=configs)
    merged_df["msd"] = pd.Series(msd_full_dict)

    # Compute RDF first peak feature.
    rdf_peak_dict = {}
    for config in configs:
        rdf_df = rdf_data_dict.get(config)
        if rdf_df is not None:
            rdf_peak_dict[config] = get_first_rdf_peak_height(rdf_df)
        else:
            rdf_peak_dict[config] = np.nan
    merged_df["rdf"] = pd.Series(rdf_peak_dict)

    # Compute hexatic order parameter (psi6) using batch analysis.
    df_hexatic = batch_hexatic(base_dir)
    psi6_dict = df_hexatic.set_index("config")["psi6"].to_dict() if not df_hexatic.empty else {}
    merged_df["psi6"] = pd.Series(psi6_dict)

    # Compute additional RDF-based features.
    cn_at_1_5_dict = {}
    slope_dict = {}
    for config in configs:
        rdf_df = rdf_data_dict.get(config)
        if rdf_df is not None:
            cn_at_1_5, overall_slope = get_coord(rdf_df)
            cn_at_1_5_dict[config] = cn_at_1_5
            slope_dict[config] = overall_slope
        else:
            cn_at_1_5_dict[config] = np.nan
            slope_dict[config] = np.nan
    merged_df["cn_at_1.5"] = pd.Series(cn_at_1_5_dict)
    merged_df["cn_slope"] = pd.Series(slope_dict)

    # Compute ratio features.
    merged_df["rdf/msd"] = merged_df["rdf"] / merged_df["msd"]
    merged_df["psi6/msd"] = merged_df["psi6"] / merged_df["msd"]
    merged_df["cn_1.5/msd"] = merged_df["cn_at_1.5"] / merged_df["msd"]
    merged_df["cn_slope/msd"] = merged_df["cn_slope"] / merged_df["msd"]

    # Reset index to include config as a column.
    merged_df = merged_df.reset_index().rename(columns={"index": "config"})

    # Write the merged features to a CSV file.
    output_csv = "features.csv"
    merged_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
