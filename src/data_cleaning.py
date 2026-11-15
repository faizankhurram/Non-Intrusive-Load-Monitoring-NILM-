"""
Clean the filtered dataset:
  - Parse timestamps
  - Sort by home + time
  - Remove duplicates
  - Reindex every home onto a continuous 1-minute timeline
  - Forward-fill short gaps; longer gaps stay NaN (windows skip them)
  - Build the whole-house aggregate: grid + solar + solar2
  - Clip negative readings to zero, convert kW -> W
"""
import os
import pandas as pd
import numpy as np
from src.config import (
    PROCESSED_DIR, TARGET_APPLIANCES, MAX_GAP_FILL, SOLAR_COLUMNS
)


def _reindex_home(home_df: pd.DataFrame) -> pd.DataFrame:
    home_id = home_df["dataid"].iloc[0]
    home_df = home_df.set_index("localminute")
    full = pd.date_range(home_df.index[0], home_df.index[-1], freq="min")
    home_df = home_df.reindex(full)
    home_df.index.name = "localminute"
    home_df["dataid"] = home_id
    return home_df.reset_index()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    df = df.copy()

    df["localminute"] = pd.to_datetime(df["localminute"], utc=True)
    df = df.sort_values(["dataid", "localminute"]).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["dataid", "localminute"], keep="first")
    print(f"[cleaning] Removed {before - len(df)} duplicate rows")

    # Continuous 1-minute timeline per home so windows span exactly 60 minutes
    before = len(df)
    df = pd.concat([_reindex_home(h) for _, h in df.groupby("dataid")], ignore_index=True)
    print(f"[cleaning] Inserted {len(df) - before:,} missing minutes")

    numeric_cols = ["grid"] + SOLAR_COLUMNS + TARGET_APPLIANCES
    df[numeric_cols] = df.groupby("dataid")[numeric_cols].transform(
        lambda s: s.ffill(limit=MAX_GAP_FILL)
    )

    # Whole-house consumption: grid is net of solar generation, so add it back.
    # Homes without solar have NaN solar columns -> treated as 0.
    solar = df[SOLAR_COLUMNS].fillna(0.0).sum(axis=1)
    df["aggregate"] = (df["grid"] + solar).clip(lower=0.0)

    n_gap = int(df["aggregate"].isna().sum())
    print(f"[cleaning] {n_gap:,} minutes remain unfilled (gaps > {MAX_GAP_FILL} min); "
          f"windows touching them are skipped")

    for col in TARGET_APPLIANCES:
        df[col] = df[col].fillna(0.0).clip(lower=0.0)

    # Data is in kW – convert to watts
    for col in ["aggregate"] + TARGET_APPLIANCES:
        df[col] = df[col] * 1000.0

    df = df[["dataid", "localminute", "aggregate"] + TARGET_APPLIANCES]
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
