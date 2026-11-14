"""
Main pipeline: runs the entire NILM project end-to-end.
Supports pause/resume — press Ctrl+C once during training to stop
gracefully. Re-run this script to resume where you left off.

Usage: python run_pipeline.py
"""
import sys
import os
import json

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TARGET_APPLIANCES, SEED, MODELS_DIR, RESULTS_DIR, WINDOWS_DIR
from src.data_loader import load_and_filter, save_filtered, load_filtered
from src.data_cleaning import clean, save_cleaned, load_cleaned
from src.labeling import compute_thresholds, apply_labels, print_label_stats
from src.splitting import chronological_split
from src.windowing import create_windows, save_windows, load_windows
from src.dataset import build_datasets
from src.models.mlp import NILM_MLP
from src.models.lstm_model import NILM_LSTM
from src.training import train_model
from src.evaluate import evaluate_model
from src.visualization import generate_all_plots


def _model_done(model_tag: str, appliance: str) -> bool:
    """Check if a model has already been fully trained and evaluated."""
    metrics_path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_metrics.json")
    model_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance}.pt")
    return os.path.exists(metrics_path) and os.path.exists(model_path)


def _has_checkpoint(model_tag: str, appliance: str) -> bool:
    """Check if an incomplete training checkpoint exists."""
    return os.path.exists(os.path.join(MODELS_DIR, f"{model_tag}_{appliance}_ckpt.pt"))


def _windows_exist() -> bool:
    """Check if windowed data already exists."""
    return all(
        os.path.exists(os.path.join(WINDOWS_DIR, f"{s}.npz"))
        for s in ["train", "val", "test"]
    )


def main():
    print("=" * 60)
    print("NILM ANN Project - Full Pipeline")
    print("=" * 60)

    # Steps 1-5: Data preparation (skip if windows already cached)
    if _windows_exist():
        print("\n[resume] Windowed data found — skipping data preparation steps 1-5")
    else:
        print("\n--- Step 1: Loading raw data ---")
        df = load_and_filter()
        save_filtered(df)

        print("\n--- Step 2: Cleaning data ---")
        df = clean(df)
        save_cleaned(df)

        print("\n--- Step 3: Splitting data ---")
        splits = chronological_split(df)

        print("\n--- Step 4: Computing thresholds and labels ---")
        thresholds = compute_thresholds(splits["train"])
        print(f"Thresholds (watts): {thresholds}")
        for name in ["train", "val", "test"]:
            splits[name] = apply_labels(splits[name], thresholds)
            print_label_stats(splits[name], name)

        print("\n--- Step 5: Creating windows ---")
        for name in ["train", "val", "test"]:
            windows = create_windows(splits[name], name)
            save_windows(windows, name)

    # Step 6: Train and evaluate models for each appliance
    all_metrics = []
    paused = False

    for idx, appliance in enumerate(TARGET_APPLIANCES):
        print(f"\n{'=' * 60}")
        print(f"Appliance {idx + 1}/{len(TARGET_APPLIANCES)}: {appliance}")
        print("=" * 60)

        train_ds, val_ds, test_ds, scaler = build_datasets(idx)

        for model_tag, ModelClass in [("mlp", NILM_MLP), ("lstm", NILM_LSTM)]:
            tag_upper = model_tag.upper()

            if _model_done(model_tag, appliance):
                # Already completed — load saved metrics
                with open(os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_metrics.json")) as f:
                    metrics = json.load(f)
                all_metrics.append(metrics)
                print(f"\n  [{tag_upper}] {appliance}: already done (loaded saved results)")
                continue

            print(f"\n--- Training {tag_upper} for {appliance} ---")
            if _has_checkpoint(model_tag, appliance):
                print(f"  [resume] Found checkpoint — continuing training")

            model = ModelClass()
            try:
                model, _ = train_model(model, train_ds, val_ds, appliance, model_tag)
                metrics = evaluate_model(model, test_ds, appliance, model_tag)
                all_metrics.append(metrics)
            except KeyboardInterrupt:
                print(f"\n  Pipeline paused. Run 'python run_pipeline.py' to resume.")
                paused = True
                break

        if paused:
            break

    if paused:
        return

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
