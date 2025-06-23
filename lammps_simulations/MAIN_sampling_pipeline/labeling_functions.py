#!/usr/bin/env python3

import numpy as np
import pandas as pd
from snorkel.labeling import labeling_function

# === Label values ===
STABLE, UNSTABLE, ABSTAIN = 1, 0, -1

# === Manually inserted 5th/95th percentiles ===
THRESHOLDS = {
    'msd':         (np.float64(0.00020689859008544), np.float64(0.000582527971282)),
    'rdf':         (np.float64(18.31646),              np.float64(31.487779999999994)),
    'psi6':        (np.float64(0.37072653206794076),   np.float64(0.4941995748061643)),
    'rdf/msd':     (np.float64(32026.940024157167),   np.float64(135801.36816802842)),
    'psi/msd':     (np.float64(658.2686962381362),    np.float64(2378.769812283029)),
    'cn_at_1.5':   (np.float64(10.045),               np.float64(15.491679999999999)),
    'cn_slope':    (np.float64(16.802403792209585),   np.float64(29.261021985298708)),
    'cn_1.5/msd':  (np.float64(17577.69815863569),    np.float64(66194.56692263029)),
    'cn_slope/msd':(np.float64(31446.753560420468),   np.float64(142069.65268061784)),
}


def _clean_name(feature: str) -> str:
    """Sanitize a feature name into a valid LF name."""
    # replace punctuation with underscores
    return feature.replace('/', '_').replace('.', '_').replace('-', '_')


def make_pct_lf(feature, low, high):
    """
    Create a percentiles-based labeling function for `feature`:
      - ABSTAIN on NaN
      - STABLE if low <= value <= high
      - UNSTABLE otherwise
    """
    @labeling_function(name=f"lf_{_clean_name(feature)}_pct")
    def lf(x, feature=feature, low=low, high=high):
        val = x[feature]
        if pd.isna(val):
            return ABSTAIN
        return STABLE if (low <= val <= high) else UNSTABLE
    return lf

# === Build one LF per feature ===
LFS = [make_pct_lf(f, lo, hi) for f, (lo, hi) in THRESHOLDS.items()]
