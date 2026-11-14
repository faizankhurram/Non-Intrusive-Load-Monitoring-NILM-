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
from src.config import RESULTS_DIR, MODELS_DIR, TARGET_APPLIANCES


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
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{model_tag}_{appliance}_curves.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison():
    """Bar chart comparing MLP vs LSTM across all appliances and metrics."""
    metrics_data = []
    for tag in ["mlp", "lstm"]:
        for appliance in TARGET_APPLIANCES:
            path = os.path.join(RESULTS_DIR, f"{tag}_{appliance}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    metrics_data.append(json.load(f))

    if not metrics_data:
        print("[viz] No metrics found to compare.")
        return

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metric_names):
        mlp_vals, lstm_vals = [], []
        labels = []
        for appliance in TARGET_APPLIANCES:
            for m in metrics_data:
                if m["appliance"] == appliance and m["model"] == "mlp":
                    mlp_vals.append(m[metric])
                if m["appliance"] == appliance and m["model"] == "lstm":
                    lstm_vals.append(m[metric])
            labels.append(appliance)

        x = np.arange(len(labels))
        width = 0.35
        if mlp_vals:
            ax.bar(x - width/2, mlp_vals, width, label="MLP")
        if lstm_vals:
            ax.bar(x + width/2, lstm_vals, width, label="LSTM")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1.05)

    fig.suptitle("MLP vs LSTM Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots():
    """Generate every required plot."""
    for tag in ["mlp", "lstm"]:
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
