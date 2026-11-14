"""
Create sliding input windows from the time-series data.
Each window: WINDOW_SIZE consecutive 'grid' readings -> predict
appliance state and power at the last timestep.
Windows do NOT cross home boundaries.
"""
import numpy as np
import pandas as pd
import os
from src.config import WINDOW_SIZE, TARGET_APPLIANCES, WINDOWS_DIR


def create_windows(df: pd.DataFrame, split_name: str) -> dict:
    """
    Build windows for one split (train / val / test).
    Returns dict with:
      'X': (N, WINDOW_SIZE) grid windows
      'y_class': (N, num_appliances) binary labels
      'y_power': (N, num_appliances) appliance power values
    """
    all_X, all_y_class, all_y_power = [], [], []

    for home_id in df["dataid"].unique():
        home_df = df[df["dataid"] == home_id].sort_values("localminute")
        grid_vals = home_df["grid"].values

        label_cols = [f"{a}_label" for a in TARGET_APPLIANCES]
        labels = home_df[label_cols].values
        powers = home_df[TARGET_APPLIANCES].values

        n = len(grid_vals)
        if n < WINDOW_SIZE:
            continue

        for i in range(WINDOW_SIZE, n):
            window = grid_vals[i - WINDOW_SIZE: i]
            all_X.append(window)
            all_y_class.append(labels[i])
            all_y_power.append(powers[i])

    X = np.array(all_X, dtype=np.float32)
    y_class = np.array(all_y_class, dtype=np.float32)
    y_power = np.array(all_y_power, dtype=np.float32)

    print(f"[windowing] {split_name}: X={X.shape}  "
          f"y_class={y_class.shape}  y_power={y_power.shape}")

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
