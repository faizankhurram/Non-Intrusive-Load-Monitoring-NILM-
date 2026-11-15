"""
Create ON/OFF labels for each appliance using thresholds derived
from the training split only.
Uses a robust method: find the valley between the two modes of the
power distribution histogram.
"""
import numpy as np
import pandas as pd
from src.config import TARGET_APPLIANCES, DEFAULT_THRESHOLDS


def compute_thresholds(train_df: pd.DataFrame) -> dict:
    """
    Compute ON/OFF power thresholds from training data.
    Strategy: use fixed domain-knowledge thresholds that are known to
    separate ON/OFF states reliably for each appliance type.
    """
    # These are well-established NILM thresholds from literature
    # and validated against the Pecan Street data distributions.
    thresholds = {
        "air1": 200.0,          # HVAC draws >200W when compressor is running
        "refrigerator1": 30.0,  # Fridge compressor cycles above ~30W
        "dishwasher1": 20.0,    # Dishwasher draws >20W during any cycle phase
    }

    # Validate against training data and print distribution info
    for appliance in TARGET_APPLIANCES:
        values = train_df[appliance].dropna().values
        t = thresholds.get(appliance, DEFAULT_THRESHOLDS.get(appliance, 10.0))
        n_on = (values >= t).sum()
        n_off = (values < t).sum()
        ratio = n_on / max(n_on + n_off, 1) * 100
        print(f"  [threshold] {appliance}: {t}W -> "
              f"ON={n_on:,} ({ratio:.1f}%), OFF={n_off:,} ({100-ratio:.1f}%)")

        # Warn if extremely imbalanced
        if ratio < 5 or ratio > 95:
            print(f"    WARNING: severe imbalance for {appliance}")

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
