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

from src.config import (
    TARGET_APPLIANCES, SEED, MODELS_DIR, RESULTS_DIR, WINDOWS_DIR,
    MODEL_TAGS, DATA_SIGNATURE,
)
from src.data_loader import load_and_filter, save_filtered, load_filtered
from src.data_cleaning import clean, save_cleaned, load_cleaned
from src.labeling import compute_thresholds, apply_labels, print_label_stats
from src.splitting import chronological_split
from src.windowing import create_windows, save_windows, load_windows
from src.dataset import build_datasets
from src.models.mlp import NILM_MLP
from src.models.lstm_model import NILM_LSTM
from src.models.gru_model import NILM_GRU
from src.models.transformer_model import NILM_Transformer
from src.training import train_model
from src.evaluate import evaluate_model
from src.visualization import generate_all_plots

MODEL_CLASSES = {
    "mlp": NILM_MLP,
    "lstm": NILM_LSTM,
    "gru": NILM_GRU,
    "transformer": NILM_Transformer,
}
SIGNATURE_PATH = os.path.join(WINDOWS_DIR, "signature.txt")


def _model_done(model_tag: str, appliance: str) -> bool:
    """Check if a model has been fully trained and evaluated on the current data."""
    metrics_path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_metrics.json")
    model_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance}.pt")
    if not (os.path.exists(metrics_path) and os.path.exists(model_path)):
        return False
    with open(metrics_path) as f:
        return json.load(f).get("data_signature") == DATA_SIGNATURE


def _has_checkpoint(model_tag: str, appliance: str) -> bool:
    """Check if an incomplete training checkpoint exists."""
    return os.path.exists(os.path.join(MODELS_DIR, f"{model_tag}_{appliance}_ckpt.pt"))


def _windows_exist() -> bool:
    """Check if windowed data for the current configuration already exists."""
    if not all(os.path.exists(os.path.join(WINDOWS_DIR, f"{s}.npz"))
               for s in ["train", "val", "test"]):
        return False
    if not os.path.exists(SIGNATURE_PATH):
        return False
    with open(SIGNATURE_PATH) as f:
        return f.read().strip() == DATA_SIGNATURE


def _clear_stale_checkpoints():
    """Checkpoints from a different data configuration cannot be resumed."""
    for name in os.listdir(MODELS_DIR):
        if name.endswith("_ckpt.pt"):
            os.remove(os.path.join(MODELS_DIR, name))
            print(f"  [rebuild] Removed stale checkpoint {name}")


def main():
    print("=" * 60)
    print("NILM ANN Project - Full Pipeline")
    print("=" * 60)

    # Steps 1-5: Data preparation (skip if windows already cached)
    if _windows_exist():
        print("\n[resume] Windowed data found — skipping data preparation steps 1-5")
    else:
        _clear_stale_checkpoints()
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
        with open(SIGNATURE_PATH, "w") as f:
            f.write(DATA_SIGNATURE)

    # Step 6: Train and evaluate models for each appliance
    all_metrics = []
    paused = False

    for idx, appliance in enumerate(TARGET_APPLIANCES):
        print(f"\n{'=' * 60}")
        print(f"Appliance {idx + 1}/{len(TARGET_APPLIANCES)}: {appliance}")
        print("=" * 60)

        train_ds, val_ds, test_ds, scaler = build_datasets(idx)

        for model_tag in MODEL_TAGS:
            ModelClass = MODEL_CLASSES[model_tag]
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
    print(f"{'Model':<12} {'Appliance':<16} {'Acc':>8} {'Bal Acc':>8} "
          f"{'F1':>8} {'MAE':>8} {'RMSE':>8}  Benchmark")
    print("-" * 80)
    for m in all_metrics:
        passed = (m["accuracy"] >= 0.85 and m["balanced_accuracy"] >= 0.75
                  and m["macro_f1"] >= 0.70)
        print(f"{m['model']:<12} {m['appliance']:<16} {m['accuracy']:>8.4f} "
              f"{m['balanced_accuracy']:>8.4f} {m['macro_f1']:>8.4f} "
              f"{m['power_mae']:>8.2f} {m['power_rmse']:>8.2f}  "
              f"{'PASS' if passed else 'FAIL'}")
    print("Benchmark = Acc>=0.85, BalAcc>=0.75, Macro-F1>=0.70 (CCP minimum)")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
