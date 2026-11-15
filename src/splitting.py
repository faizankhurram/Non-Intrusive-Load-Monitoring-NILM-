"""
Chronological train / validation / test splitting.
Every home is split chronologically — no full-home holdout, so that
the test set has representative appliance distributions from all homes.
"""
import pandas as pd
from src.config import (
    TRAIN_RATIO, VAL_RATIO, SELECTED_HOMES
)


def chronological_split(df: pd.DataFrame) -> dict:
    """
    Split data chronologically per home.
    Every home contributes to train, val, and test.
    Returns dict with keys 'train', 'val', 'test'.
    """
    train_parts, val_parts, test_parts = [], [], []

    for home_id in SELECTED_HOMES:
        home_df = df[df["dataid"] == home_id].sort_values("localminute")

        n = len(home_df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train_parts.append(home_df.iloc[:train_end])
        val_parts.append(home_df.iloc[train_end:val_end])
        test_parts.append(home_df.iloc[val_end:])

        print(f"[split] Home {home_id}: train={train_end:,}  "
              f"val={val_end - train_end:,}  test={n - val_end:,}")

    splits = {
        "train": pd.concat(train_parts, ignore_index=True),
        "val": pd.concat(val_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }

    for name, part in splits.items():
        print(f"[split] Total {name}: {len(part):,} rows")

    return splits
