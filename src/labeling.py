"""ON/OFF labeling from fixed appliance power thresholds."""
import numpy as np
import pandas as pd
from src.config import TARGET_APPLIANCES, DEFAULT_THRESHOLDS


def compute_thresholds(train_df: pd.DataFrame) -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)

    for appliance in TARGET_APPLIANCES:
        values = train_df[appliance].dropna().values
        t = thresholds.get(appliance, 10.0)
        n_on = (values >= t).sum()
        n_off = (values < t).sum()
        ratio = n_on / max(n_on + n_off, 1) * 100
        print(f"  [threshold] {appliance}: {t}W -> "
              f"ON={n_on:,} ({ratio:.1f}%), OFF={n_off:,} ({100-ratio:.1f}%)")
        if ratio < 5 or ratio > 95:
            print(f"    WARNING: severe imbalance for {appliance}")

    return thresholds


def apply_labels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    df = df.copy()
    for appliance in TARGET_APPLIANCES:
        label_col = f"{appliance}_label"
        t = thresholds[appliance]
        df[label_col] = (df[appliance] >= t).astype(np.int64)
    return df


def print_label_stats(df: pd.DataFrame, split_name: str = ""):
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
