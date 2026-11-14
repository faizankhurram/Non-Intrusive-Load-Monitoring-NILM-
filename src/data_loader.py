"""
Load raw Pecan Street CSV and filter to selected homes / columns.
Saves a compact parquet file for fast subsequent access.
"""
import os
import pandas as pd
from src.config import (
    RAW_CSV, PROCESSED_DIR, SELECTED_HOMES, KEEP_COLUMNS
)


def load_and_filter(chunk_size: int = 500_000) -> pd.DataFrame:
    """Read the large raw CSV in chunks, keep only selected homes/columns."""
    frames = []
    for chunk in pd.read_csv(
        RAW_CSV,
        usecols=KEEP_COLUMNS,
        chunksize=chunk_size,
        low_memory=False,
    ):
        filtered = chunk[chunk["dataid"].isin(SELECTED_HOMES)]
        if not filtered.empty:
            frames.append(filtered)
    df = pd.concat(frames, ignore_index=True)
    print(f"[data_loader] Loaded {len(df):,} rows for {df['dataid'].nunique()} homes")
    return df


def save_filtered(df: pd.DataFrame) -> str:
    """Save filtered dataframe to parquet."""
    path = os.path.join(PROCESSED_DIR, "filtered_austin.parquet")
    df.to_parquet(path, index=False)
    print(f"[data_loader] Saved filtered data to {path}")
    return path


def load_filtered() -> pd.DataFrame:
    """Load previously saved filtered data."""
    path = os.path.join(PROCESSED_DIR, "filtered_austin.parquet")
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = load_and_filter()
    save_filtered(df)
