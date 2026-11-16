"""Create sliding windows of whole-house power for each split."""
import numpy as np
import pandas as pd
import os
from numpy.lib.stride_tricks import sliding_window_view
from src.config import WINDOW_SIZE, WINDOW_STEP, TARGET_APPLIANCES, WINDOWS_DIR


def create_windows(df: pd.DataFrame, split_name: str) -> dict:
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

    X = np.concatenate(all_X) if all_X else np.empty((0, WINDOW_SIZE), dtype=np.float32)
    y_class = np.concatenate(all_y_class) if all_y_class else np.empty((0, len(TARGET_APPLIANCES)))
    y_power = np.concatenate(all_y_power) if all_y_power else np.empty((0, len(TARGET_APPLIANCES)))

    print(f"[windowing] {split_name}: X={X.shape}  y_class={y_class.shape}  "
          f"y_power={y_power.shape}  (skipped {skipped} windows overlapping gaps)")
    return {"X": X, "y_class": y_class, "y_power": y_power}


def save_windows(windows: dict, split_name: str):
    path = os.path.join(WINDOWS_DIR, f"{split_name}.npz")
    np.savez_compressed(path, **windows)
    print(f"[windowing] Saved {split_name} windows to {path}")


def load_windows(split_name: str) -> dict:
    path = os.path.join(WINDOWS_DIR, f"{split_name}.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}
