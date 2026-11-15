"""
Generate all required plots:
  - Confusion matrices
  - Actual vs predicted power
  - Training loss curves
  - Model comparison bar charts
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import RESULTS_DIR, MODELS_DIR, TARGET_APPLIANCES, MODEL_TAGS


def plot_confusion_matrix(appliance: str, model_tag: str):
    """Plot and save confusion matrix heatmap."""
    path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_tag.upper()} - {appliance} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_cm.png"), dpi=150)
    plt.close(fig)


def plot_power_prediction(appliance: str, model_tag: str, n_points: int = 500):
    """Plot actual vs predicted power for a slice of test data."""
    pred_path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_preds.npz")
    data = np.load(pred_path)
    pw_true = data["pow_true"][:n_points]
    pw_pred = data["pow_pred"][:n_points]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pw_true, label="Actual", alpha=0.8)
    ax.plot(pw_pred, label="Predicted", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"{model_tag.upper()} - {appliance} Power: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_power.png"), dpi=150)
    plt.close(fig)


def plot_training_curves(appliance: str, model_tag: str):
    """Plot training and validation loss curves."""
    hist_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance}_history.json")
    with open(hist_path) as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_tag.upper()} - {appliance} Training Curves")
    if "val_f1" in history:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["val_f1"], color="tab:green", linestyle="--",
                 label="Val Macro-F1")
        ax2.set_ylabel("Val Macro-F1")
        ax2.set_ylim(0, 1)
        lines = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="center right")
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_curves.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison():
    """Bar chart comparing all models across all appliances and metrics."""
    metrics = {}
    for tag in MODEL_TAGS:
        for appliance in TARGET_APPLIANCES:
            path = os.path.join(RESULTS_DIR, f"{tag}_{appliance}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    metrics[(tag, appliance)] = json.load(f)

    if not metrics:
        print("[viz] No metrics found to compare.")
        return

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    minimums = {"accuracy": 0.85, "balanced_accuracy": 0.75, "macro_f1": 0.70}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(TARGET_APPLIANCES))
    width = 0.8 / len(MODEL_TAGS)

    for ax, metric in zip(axes, metric_names):
        for i, tag in enumerate(MODEL_TAGS):
            vals = [metrics.get((tag, a), {}).get(metric, 0.0) for a in TARGET_APPLIANCES]
            offset = (i - (len(MODEL_TAGS) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=tag.upper())
        ax.axhline(minimums[metric], color="red", linestyle="--", linewidth=1,
                   label="CCP minimum")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_APPLIANCES, rotation=15)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Model Comparison: " + " vs ".join(t.upper() for t in MODEL_TAGS))
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots():
    """Generate every required plot."""
    for tag in MODEL_TAGS:
        for appliance in TARGET_APPLIANCES:
            try:
                plot_confusion_matrix(appliance, tag)
                plot_power_prediction(appliance, tag)
                plot_training_curves(appliance, tag)
            except FileNotFoundError:
                print(f"[viz] Skipping {tag}/{appliance} - files not found")
    plot_model_comparison()
    print("[viz] All plots saved to", RESULTS_DIR)


if __name__ == "__main__":
    generate_all_plots()
