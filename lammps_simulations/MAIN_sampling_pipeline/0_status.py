#!/usr/bin/env python3
"""
check_progress.py

Walk through every sub‑directory whose name starts with “config_”
and count how many of them already contain the various milestone
files produced by the workflow.

Outputs a short summary like

    Total config dirs          : 256
      ├─ with system_after_min.data        : 256
      ├─ with system_after_short_nvt.data  : 255
      ├─ with full_traj_long.lammpstrj     : 240
      └─ with system_after_nvt_long.data   : 240
"""

import sys
from pathlib import Path

# file names we care about (folder / filename)
MILESTONES = [
    "system_after_min.data",
    "system_after_short_nvt.data",
    "full_traj_long.lammpstrj",
    "system_after_nvt_long.data",
]

def main(base: Path):
    config_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("config_"))

    counts = {name: 0 for name in MILESTONES}

    for cfg in config_dirs:
        for fname in MILESTONES:
            if (cfg / fname).is_file():
                counts[fname] += 1

    # ---- print summary -------------------------------------------------
    print(f"Total config dirs : {len(config_dirs)}")
    print("  ├─ with system_after_min.data        :", counts['system_after_min.data'])
    print("  ├─ with system_after_short_nvt.data  :", counts['system_after_short_nvt.data'])
    print("  ├─ with full_traj_long.lammpstrj     :", counts['full_traj_long.lammpstrj'])
    print("  └─ with system_after_nvt_long.data   :", counts['system_after_nvt_long.data'])

if __name__ == "__main__":
    base_dir = Path(".") if len(sys.argv) == 1 else Path(sys.argv[1])
    if not base_dir.is_dir():
        sys.exit(f"Error: {base_dir} is not a directory")
    main(base_dir)
