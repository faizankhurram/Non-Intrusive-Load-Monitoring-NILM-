"""
Create ON/OFF labels for each appliance using thresholds derived
from the training split only.
"""
import numpy as np
import pandas as pd
from src.config import TARGET_APPLIANCES, DEFAULT_THRESHOLDS


def compute_thresholds(train_df: pd.DataFrame) -> dict:
    """
    Compute ON/OFF power thresholds from training data.
    Strategy: use the midpoint between the two modes of the power
    distribution (bimodal split). Falls back to DEFAULT_THRESHOLDS
    if the histogram method fails.
    """
    thresholds = {}
    for appliance in TARGET_APPLIANCES:
        values = train_df[appliance].dropna().values
        if len(values) == 0:
            thresholds[appliance] = DEFAULT_THRESHOLDS[appliance]
            continue

        # Use percentile-based approach: the 10th percentile of
        # non-zero readings as the threshold
        nonzero = values[values > 0]
        if len(nonzero) > 100:
            threshold = np.percentile(nonzero, 10)
            # Ensure a reasonable minimum
            threshold = max(threshold, DEFAULT_THRESHOLDS.get(appliance, 5.0))
        else:
            threshold = DEFAULT_THRESHOLDS[appliance]

        thresholds[appliance] = round(float(threshold), 2)
    return thresholds


def apply_labels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add binary ON/OFF label columns for each appliance."""
    df = df.copy()
    for appliance in TARGET_APPLIANCES:
        label_col = f"{appliance}_label"
        t = thresholds[appliance]
        df[label_col] = (df[appliance] >= t).astype(np.int64)
    return df


def print_label_stats(df: pd.DataFrame, split_name: str = ""):
    """Print ON/OFF counts for each appliance."""
    prefix = f"[{split_name}] " if split_name else ""
    for appliance in TARGET_APPLIANCES:
        label_col = f"{appliance}_label"
        if label_col in df.columns:
            counts = df[label_col].value_counts().to_dict()
            total = len(df)
            on = counts.get(1, 0)
            off = counts.get(0, 0)
            print(f"{prefix}{appliance}: ON={on:,} ({100*on/total:.1f}%)  "
                  f"OFF={off:,} ({100*off/total:.1f}%)")
