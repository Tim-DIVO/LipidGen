#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#############################
#   Utility Functions       #
#############################

def parse_lammps_dump_for_com(dump_file: str, atom_type: int = 1):
    """
    A rudimentary parser for a LAMMPS custom dump file.
    Assumes that frames are delimited by lines starting with "ITEM: TIMESTEP"
    and that each frame contains an "ITEM: ATOMS" section.
    Returns the COM of atoms (of the specified type) for the first and last frames.
    """
    with open(dump_file, "r") as f:
        lines = f.readlines()
    
    frames = []
    current_frame = []
    for line in lines:
        if line.startswith("ITEM: TIMESTEP"):
            if current_frame:
                frames.append(current_frame)
            current_frame = []
        else:
            current_frame.append(line.strip())
    if current_frame:
        frames.append(current_frame)
    
    if len(frames) < 2:
        raise ValueError(f"Not enough frames in {dump_file} to compute COM drift.")
    
    def get_com(frame_lines):
        positions = []
        start_collect = False
        for ln in frame_lines:
            if ln.startswith("ITEM: ATOMS"):
                start_collect = True
                continue
            if start_collect:
                parts = ln.split()
                # Assume format: id type x y z ...
                if int(parts[1]) == atom_type:
                    pos = [float(parts[2]), float(parts[3]), float(parts[4])]
                    positions.append(pos)
        if not positions:
            raise ValueError("No atoms of the specified type found in frame.")
        return np.mean(np.array(positions), axis=0)
    
    com_first = get_com(frames[0])
    com_last  = get_com(frames[-1])
    return np.array(com_first), np.array(com_last)

def feature_com_drift(dump_file: str, atom_type: int = 1) -> float:
    """Compute the COM drift from first to last frame."""
    com1, com2 = parse_lammps_dump_for_com(dump_file, atom_type=atom_type)
    return np.linalg.norm(com2 - com1)

def parse_neighbors_over_time(neighbor_file: str) -> np.ndarray:
    """Load neighbor data (assumed two columns: step and avg_neighbors)."""
    return np.loadtxt(neighbor_file)

def feature_neighbor_drop(neighbor_file: str) -> float:
    """Return the drop (initial - final) in neighbor count."""
    data = parse_neighbors_over_time(neighbor_file)
    if data.ndim == 1 or data.shape[0] < 2:
        return np.nan
    initial = data[0, 1] if data.ndim > 1 else data[0]
    final   = data[-1, 1] if data.ndim > 1 else data[-1]
    return initial - final

def parse_msd(msd_file: str) -> np.ndarray:
    """Load MSD data assumed to have two columns: time and msd."""
    return np.loadtxt(msd_file)

def feature_msd_plateau(msd_file: str, fraction: float = 0.3) -> float:
    """Compute the slope of MSD in the final fraction of the simulation."""
    arr = parse_msd(msd_file)
    if arr.ndim < 2 or arr.shape[0] < 5:
        return np.nan
    n = arr.shape[0]
    start_idx = int((1 - fraction) * n)
    times = arr[start_idx:, 0]
    msd_vals = arr[start_idx:, 1]
    A = np.vstack([times, np.ones(len(times))]).T
    slope, _ = np.linalg.lstsq(A, msd_vals, rcond=None)[0]
    return slope

def parse_rdf_file(filename: str) -> dict:
    """
    Parse an rdf file (assumed to be in blocks).
    Each block starts with a header line containing "TimeStep" and a number of rows,
    followed by that many rows of data (columns: index, radius, g(r1), c(1), g(r2), c(2), g(r3), c(3), g(r4), c(4)).
    Returns a dictionary mapping timestep (float) -> DataFrame.
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
                row = [float(tok) for tok in lines[i].split()]
                block_data.append(row)
                i += 1
            col_names = ["index", "radius", "g(r1)", "c(1)", "g(r2)", "c(2)", "g(r3)", "c(3)", "g(r4)", "c(4)"]
            df = pd.DataFrame(block_data, columns=col_names)
            blocks[timestep] = df
        else:
            i += 1
    return blocks

def feature_rdf_sharpness(rdf_file: str) -> float:
    """
    Compute a measure of RDF sharpness.
    Here we take the first peak's height relative to a baseline (average g(r) for r>2.0).
    """
    blocks = parse_rdf_file(rdf_file)
    if not blocks:
        return np.nan
    # Select the block corresponding to the final timestep
    final_ts = max(blocks.keys())
    df = blocks[final_ts]
    r = df["radius"].values
    g = df["g(r1)"].values  # for example, use the first pair's RDF
    if len(r) == 0:
        return np.nan
    mask = r > 2.0
    baseline = np.mean(g[mask]) if np.any(mask) else np.nan
    peak = np.max(g)
    return (peak - baseline) / (baseline + 1e-9)

def read_lammps_data_positions(data_file: str, atom_type: int = 1) -> np.ndarray:
    """
    Very basic parser for a LAMMPS data file.
    Looks for a line starting with "Atoms" and then reads all subsequent lines
    until an empty line or a new section header.
    Assumes each line in the Atoms section has: id mol type charge x y z.
    Returns an array of positions for atoms of the given type.
    """
    positions = []
    with open(data_file, "r") as f:
        lines = f.readlines()
    
    in_atoms_section = False
    for line in lines:
        if line.strip().startswith("Atoms"):
            in_atoms_section = True
            continue
        if in_atoms_section:
            if line.strip() == "" or line.startswith("Bonds"):
                break
            parts = line.split()
            if int(parts[2]) == atom_type:
                pos = [float(parts[4]), float(parts[5]), float(parts[6])]
                positions.append(pos)
    return np.array(positions)

def feature_minimization_dispersion(min_file: str, atom_type: int = 1) -> float:
    """
    Compute the dispersion of head bead positions (e.g., type 1) in the minimized configuration.
    We calculate the standard deviation of the radial distances from the COM.
    A low dispersion suggests that the sphere remains intact.
    """
    positions = read_lammps_data_positions(min_file, atom_type=atom_type)
    if positions.size == 0:
        return np.nan
    com = positions.mean(axis=0)
    radii = np.linalg.norm(positions - com, axis=1)
    return np.std(radii)

def feature_cholesterol_distribution(traj_file: str, atom_type: int = 9) -> dict:
    """
    Analyze the final frame of the trajectory to assess cholesterol (type 9)
    distribution between inner and outer leaflets.
    
    Uses KMeans clustering on the radial distances of cholesterol head positions.
    Returns a dictionary with inner count, outer count, and the ratio.
    """
    with open(traj_file, "r") as f:
        lines = f.readlines()
    
    # A very simplified parser: assume the last frame starts with "ITEM: TIMESTEP"
    # and then "ITEM: ATOMS" follows. Extract positions for atom_type==9.
    last_frame = []
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith("ITEM: ATOMS"):
            # Collect from here until end of file
            last_frame = lines[i+1:]
            break
    
    positions = []
    for line in last_frame:
        parts = line.split()
        if len(parts) < 5:
            continue
        if int(parts[1]) == atom_type:
            pos = [float(parts[2]), float(parts[3]), float(parts[4])]
            positions.append(pos)
    positions = np.array(positions)
    result = {"chol_inner": np.nan, "chol_outer": np.nan, "chol_ratio": np.nan}
    if positions.size == 0:
        return result
    
    # Compute the COM of cholesterol heads as a proxy for sphere center.
    center = positions.mean(axis=0)
    radial = np.linalg.norm(positions - center, axis=1).reshape(-1, 1)
    # Cluster into 2 groups (inner and outer)
    try:
        kmeans = KMeans(n_clusters=2, random_state=42).fit(radial)
    except Exception as e:
        return result
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    # Inner: lower radial distance; Outer: higher
    inner_label = np.argmin(cluster_centers)
    outer_label = np.argmax(cluster_centers)
    inner_count = np.sum(labels == inner_label)
    outer_count = np.sum(labels == outer_label)
    ratio = (outer_count - inner_count) / (outer_count + inner_count + 1e-9)
    result["chol_inner"] = inner_count
    result["chol_outer"] = outer_count
    result["chol_ratio"] = ratio
    return result

#############################
#   Aggregation Function    #
#############################

def analyze_config_folder(folder: str) -> dict:
    """
    For a given configuration folder (e.g., config_1), extract features.
    Expected files in the folder:
      - full_traj.lammpstrj (NVT trajectory)
      - neighbors.out
      - msd_adapt.out
      - rdf_adapt.out
      - system_after_min.data (minimized configuration)
    """
    feats = {"folder": folder}
    
    # 1. COM Drift from full trajectory
    dump_file = os.path.join(folder, "full_traj.lammpstrj")
    try:
        feats["com_drift"] = feature_com_drift(dump_file, atom_type=1)
    except Exception as e:
        feats["com_drift"] = np.nan
        print(f"Error in COM drift for {folder}: {e}")
    
    # 2. Neighbor Drop
    neighbor_file = os.path.join(folder, "neighbors.out")
    try:
        feats["neighbor_drop"] = feature_neighbor_drop(neighbor_file)
    except Exception as e:
        feats["neighbor_drop"] = np.nan
        print(f"Error in neighbor drop for {folder}: {e}")
    
    # 3. MSD Plateau Slope
    msd_file = os.path.join(folder, "msd_adapt.out")
    try:
        feats["msd_slope"] = feature_msd_plateau(msd_file, fraction=0.3)
    except Exception as e:
        feats["msd_slope"] = np.nan
        print(f"Error in MSD plateau for {folder}: {e}")
    
    # 4. RDF Sharpness
    rdf_file = os.path.join(folder, "rdf_adapt.out")
    try:
        feats["rdf_sharpness"] = feature_rdf_sharpness(rdf_file)
    except Exception as e:
        feats["rdf_sharpness"] = np.nan
        print(f"Error in RDF sharpness for {folder}: {e}")
    
    # 5. Minimization Dispersion (indicates disassembly)
    min_file = os.path.join(folder, "system_after_min.data")
    try:
        feats["min_dispersion"] = feature_minimization_dispersion(min_file, atom_type=1)
    except Exception as e:
        feats["min_dispersion"] = np.nan
        print(f"Error in minimization dispersion for {folder}: {e}")
    
    # 6. Cholesterol Leaflet Distribution (from final frame of full trajectory)
    try:
        chol_feats = feature_cholesterol_distribution(dump_file, atom_type=9)
        feats.update(chol_feats)
    except Exception as e:
        feats["chol_inner"] = np.nan
        feats["chol_outer"] = np.nan
        feats["chol_ratio"] = np.nan
        print(f"Error in cholesterol distribution for {folder}: {e}")
    
    return feats

#############################
#          Main             #
#############################

def main():
    # Find all folders that match "config_*"
    config_folders = sorted(glob.glob("config_*"), key=lambda x: int(re.findall(r'\d+', x)[0]))
    all_features = []
    
    for folder in config_folders:
        print(f"Analyzing {folder} ...")
        feats = analyze_config_folder(folder)
        all_features.append(feats)
    
    # Save aggregated features to a CSV file with each row corresponding to a config folder.
    if not all_features:
        print("No config folders found!")
        return
    
    keys = sorted(all_features[0].keys())
    out_csv = "aggregated_features.csv"
    with open(out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)
    
    print(f"Aggregated features saved to {out_csv}")

if __name__ == "__main__":
    main()
