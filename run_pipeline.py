"""
Main pipeline: runs the entire NILM project end-to-end.
Usage: python run_pipeline.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TARGET_APPLIANCES, SEED
from src.data_loader import load_and_filter, save_filtered
from src.data_cleaning import clean, save_cleaned
from src.labeling import compute_thresholds, apply_labels, print_label_stats
from src.splitting import chronological_split
from src.windowing import create_windows, save_windows
from src.dataset import build_datasets
from src.models.mlp import NILM_MLP
from src.models.lstm_model import NILM_LSTM
from src.training import train_model
from src.evaluate import evaluate_model
from src.visualization import generate_all_plots


def main():
    print("=" * 60)
    print("NILM ANN Project - Full Pipeline")
    print("=" * 60)

    # Step 1: Load and filter data
    print("\n--- Step 1: Loading raw data ---")
    df = load_and_filter()
    save_filtered(df)

    # Step 2: Clean
    print("\n--- Step 2: Cleaning data ---")
    df = clean(df)
    save_cleaned(df)

    # Step 3: Split chronologically
    print("\n--- Step 3: Splitting data ---")
    splits = chronological_split(df)

    # Step 4: Compute thresholds from training data and apply labels
    print("\n--- Step 4: Computing thresholds and labels ---")
    thresholds = compute_thresholds(splits["train"])
    print(f"Thresholds (watts): {thresholds}")

    for name in ["train", "val", "test"]:
        splits[name] = apply_labels(splits[name], thresholds)
        print_label_stats(splits[name], name)

    # Step 5: Create windows
    print("\n--- Step 5: Creating windows ---")
    for name in ["train", "val", "test"]:
        windows = create_windows(splits[name], name)
        save_windows(windows, name)

    # Step 6: Train and evaluate models for each appliance
    all_metrics = []

    for idx, appliance in enumerate(TARGET_APPLIANCES):
        print(f"\n{'=' * 60}")
        print(f"Appliance {idx + 1}/{len(TARGET_APPLIANCES)}: {appliance}")
        print("=" * 60)

        train_ds, val_ds, test_ds, scaler = build_datasets(idx)

        # Model A: MLP
        print(f"\n--- Training MLP for {appliance} ---")
        mlp = NILM_MLP()
        mlp, _ = train_model(mlp, train_ds, val_ds, appliance, "mlp")
        mlp_metrics = evaluate_model(mlp, test_ds, appliance, "mlp")
        all_metrics.append(mlp_metrics)

        # Model B: LSTM
        print(f"\n--- Training LSTM for {appliance} ---")
        lstm = NILM_LSTM()
        lstm, _ = train_model(lstm, train_ds, val_ds, appliance, "lstm")
        lstm_metrics = evaluate_model(lstm, test_ds, appliance, "lstm")
        all_metrics.append(lstm_metrics)

    # Step 7: Generate all plots
    print("\n--- Step 7: Generating plots ---")
    generate_all_plots()

    # Step 8: Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<8} {'Appliance':<16} {'Acc':>8} {'Bal Acc':>8} "
          f"{'F1':>8} {'MAE':>8} {'RMSE':>8}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['model']:<8} {m['appliance']:<16} {m['accuracy']:>8.4f} "
              f"{m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} "
              f"{m['power_mae']:>8.2f} {m['power_rmse']:>8.2f}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
