"""
Chronological train / validation / test splitting.
Also reserves one full home for unseen-home testing.
"""
import pandas as pd
from src.config import (
    TRAIN_RATIO, VAL_RATIO, SELECTED_HOMES, UNSEEN_HOME
)


def chronological_split(df: pd.DataFrame) -> dict:
    """
    Split data chronologically per home.
    The UNSEEN_HOME goes entirely into the test set.
    For other homes: first 70% train, next 15% val, last 15% test.
    Returns dict with keys 'train', 'val', 'test'.
    """
    train_parts, val_parts, test_parts = [], [], []

    for home_id in SELECTED_HOMES:
        home_df = df[df["dataid"] == home_id].sort_values("localminute")

        if home_id == UNSEEN_HOME:
            test_parts.append(home_df)
            print(f"[split] Home {home_id}: all {len(home_df):,} rows -> test (unseen)")
            continue

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
