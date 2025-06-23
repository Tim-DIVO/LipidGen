#!/usr/bin/env python3
"""
apply_snorkel_to_configs.py

Load a pre‑trained Snorkel LabelModel and apply it to your feature table,
then write either stable.txt or unstable.txt into each config_X folder,
and append a 'status' column both to the feature CSV and to your
original parameter_samples CSV.
"""

import os
import pickle
import pandas as pd
from snorkel.labeling import PandasLFApplier

# -------------------------------------------------------------------
# 1) Path to your pickled LabelModel
MODEL_PATH    = "label_model.pkl"

# 2) Path to your global feature CSV (must have a 'config' column)
FEATURE_CSV   = "features_short_nvt.csv"

# 3) Path to your original Sobol parameter CSV
PARAMETER_CSV = "parameter_samples_0_4096.csv"  # adjust name as needed

# 4) Import your labeling functions (defined in labeling_functions.py)
from labeling_functions import LFS
# -------------------------------------------------------------------


def load_label_model(path):
    """Load the pickled Snorkel LabelModel."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LabelModel not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_features(path):
    """Load the feature table into a DataFrame."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    df = pd.read_csv(path)
    if "config" not in df.columns:
        raise ValueError("Feature CSV must contain a 'config' column")
    return df


def apply_model(label_model, df, lfs):
    """Apply labeling functions and the LabelModel to get predictions."""
    applier = PandasLFApplier(lfs=lfs)
    L = applier.apply(df)
    preds = label_model.predict(L=L)
    return preds


def write_markers(df, preds):
    """
    For each row in df, write stable.txt or unstable.txt into the
    corresponding config_<n> folder. Remove any existing marker first.
    """
    mapping = {1: "stable.txt", 0: "unstable.txt"}
    for cfg, pred in zip(df["config"], preds):
        folder = os.path.join(".", cfg)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found, skipping: {cfg}")
            continue
        # Remove old markers if present
        for old in ("stable.txt", "unstable.txt"):
            oldp = os.path.join(folder, old)
            if os.path.isfile(oldp):
                os.remove(oldp)
        # Write the new marker
        marker = mapping.get(pred)
        if marker:
            open(os.path.join(folder, marker), "w").close()
        else:
            print(f"[WARN] Unexpected label {pred} for {cfg}")


def main():
    # 1) Load model
    print("Loading LabelModel from", MODEL_PATH)
    label_model = load_label_model(MODEL_PATH)

    # 2) Load features
    print("Loading feature table from", FEATURE_CSV)
    df = load_features(FEATURE_CSV)

    # 3) Apply labeling functions + model
    print(f"Applying {len(LFS)} LFs to {len(df)} configs…")
    preds = apply_model(label_model, df, LFS)

    # 4) Add status column to features DataFrame
    status_map = {1: "stable", 0: "unstable"}
    df['status'] = [status_map.get(p, "unstable") for p in preds]

    # 5) Save updated feature table
    feat_out = FEATURE_CSV.replace('.csv', '_labeled.csv')
    df.to_csv(feat_out, index=False)
    print(f"Wrote updated feature table with status to {feat_out}")

    # 6) Write marker files into config folders
    print("Writing stable/unstable markers into config folders…")
    write_markers(df, preds)

    # 7) Annotate original parameter CSV
    if os.path.isfile(PARAMETER_CSV):
        print("Annotating parameter CSV", PARAMETER_CSV)
        param_df = pd.read_csv(PARAMETER_CSV)
        merged = pd.merge(param_df, df[['config', 'status']],
                          on='config', how='left')
        param_out = PARAMETER_CSV.replace('.csv', '_labeled.csv')
        merged.to_csv(param_out, index=False)
        print(f"Wrote labeled parameter CSV to {param_out}")
    else:
        print(f"[WARN] Parameter CSV not found: {PARAMETER_CSV}")

    print("Done.")

if __name__ == "__main__":
    main()

