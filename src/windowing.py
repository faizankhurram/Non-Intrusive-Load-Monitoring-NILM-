"""
Create sliding input windows from the time-series data.
Each window: WINDOW_SIZE consecutive 'aggregate' readings ending at minute t
-> predict appliance state and power at minute t (the last timestep).
Windows do NOT cross home boundaries or unfilled timestamp gaps.
"""
import numpy as np
import pandas as pd
import os
from numpy.lib.stride_tricks import sliding_window_view
from src.config import WINDOW_SIZE, WINDOW_STEP, TARGET_APPLIANCES, WINDOWS_DIR


def create_windows(df: pd.DataFrame, split_name: str) -> dict:
    """
    Build windows for one split (train / val / test).
    Returns dict with:
      'X': (N, WINDOW_SIZE) aggregate windows
      'y_class': (N, num_appliances) binary labels
      'y_power': (N, num_appliances) appliance power values
    """
    label_cols = [f"{a}_label" for a in TARGET_APPLIANCES]
    all_X, all_y_class, all_y_power = [], [], []
    skipped = 0

    for home_id in df["dataid"].unique():
        home_df = df[df["dataid"] == home_id].sort_values("localminute")
        agg = home_df["aggregate"].to_numpy(dtype=np.float32)
        if len(agg) < WINDOW_SIZE:
            continue

        windows = sliding_window_view(agg, WINDOW_SIZE)[::WINDOW_STEP]
        ends = np.arange(WINDOW_SIZE - 1, len(agg))[::WINDOW_STEP]
        valid = ~np.isnan(windows).any(axis=1)
        skipped += int((~valid).sum())

        all_X.append(windows[valid])
        all_y_class.append(home_df[label_cols].to_numpy()[ends[valid]])
        all_y_power.append(home_df[TARGET_APPLIANCES].to_numpy()[ends[valid]])

    X = np.concatenate(all_X).astype(np.float32)
    y_class = np.concatenate(all_y_class).astype(np.float32)
    y_power = np.concatenate(all_y_power).astype(np.float32)

    print(f"[windowing] {split_name}: X={X.shape}  "
          f"y_class={y_class.shape}  y_power={y_power.shape}  "
          f"(skipped {skipped:,} windows overlapping gaps)")

    return {"X": X, "y_class": y_class, "y_power": y_power}


def save_windows(windows: dict, split_name: str):
    """Save window arrays to .npz file."""
    path = os.path.join(WINDOWS_DIR, f"{split_name}.npz")
    np.savez_compressed(path, **windows)
    print(f"[windowing] Saved {split_name} windows to {path}")


def load_windows(split_name: str) -> dict:
    """Load saved window arrays."""
    path = os.path.join(WINDOWS_DIR, f"{split_name}.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}
