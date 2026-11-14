"""
Clean the filtered dataset:
  - Parse timestamps
  - Sort by home + time
  - Remove duplicates
  - Handle missing values (forward-fill short gaps)
  - Clip negative appliance readings to zero
"""
import os
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_DIR, TARGET_APPLIANCES, MAX_GAP_FILL
)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = df.copy()

    # Parse timestamp
    df["localminute"] = pd.to_datetime(df["localminute"], utc=True)
    df = df.sort_values(["dataid", "localminute"]).reset_index(drop=True)

    # Remove duplicate timestamps within each home
    before = len(df)
    df = df.drop_duplicates(subset=["dataid", "localminute"], keep="first")
    print(f"[cleaning] Removed {before - len(df)} duplicate rows")

    # Forward-fill short gaps (up to MAX_GAP_FILL minutes) per home
    numeric_cols = ["grid"] + TARGET_APPLIANCES
    df[numeric_cols] = df.groupby("dataid")[numeric_cols].transform(
        lambda s: s.ffill(limit=MAX_GAP_FILL)
    )

    # Drop rows where grid is still NaN after fill
    before = len(df)
    df = df.dropna(subset=["grid"])
    print(f"[cleaning] Dropped {before - len(df)} rows with NaN grid after fill")

    # Fill remaining NaN appliance values with 0 (appliance not drawing power)
    for col in TARGET_APPLIANCES:
        df[col] = df[col].fillna(0.0)

    # Clip negative appliance readings to zero
    for col in TARGET_APPLIANCES:
        df[col] = df[col].clip(lower=0.0)

    # Grid can go negative (solar feed-in) – clip to zero for NILM input
    df["grid"] = df["grid"].clip(lower=0.0)

    # Convert power to watts (data is already in kW – multiply by 1000)
    for col in numeric_cols:
        df[col] = df[col] * 1000.0

    print(f"[cleaning] Final shape: {df.shape}")
    return df


def save_cleaned(df: pd.DataFrame) -> str:
    path = os.path.join(PROCESSED_DIR, "cleaned_austin.parquet")
    df.to_parquet(path, index=False)
    print(f"[cleaning] Saved cleaned data to {path}")
    return path


def load_cleaned() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "cleaned_austin.parquet")
    return pd.read_parquet(path)


if __name__ == "__main__":
    from src.data_loader import load_filtered
    df = load_filtered()
    df = clean(df)
    save_cleaned(df)
