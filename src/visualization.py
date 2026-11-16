"""
Generate all required plots (one sub-folder of results/ per plot type):
  - Confusion matrices
  - Training loss curves
  - Per-class performance (normalized confusion matrix + precision/recall/F1)
  - Training dynamics (generalization gap, LR schedule, early stopping)
  - Model comparison bar charts
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    METRICS_DIR, PLOT_DIRS, MODELS_DIR,
    TARGET_APPLIANCES, MODEL_TAGS, EPOCHS,
)


def _metrics_path(model_tag: str, appliance: str) -> str:
    return os.path.join(METRICS_DIR, f"{model_tag}_{appliance}_metrics.json")


def _history_path(model_tag: str, appliance: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_tag}_{appliance}_history.json")


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(appliance: str, model_tag: str):
    """Plot and save confusion matrix heatmap."""
    cm = np.array(_load_json(_metrics_path(model_tag, appliance))["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_tag.upper()} - {appliance} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRS["confusion_matrix"], f"{model_tag}_{appliance}_cm.png"),
                dpi=150)
    plt.close(fig)


def plot_training_curves(appliance: str, model_tag: str):
    """Plot training and validation loss curves."""
    history = _load_json(_history_path(model_tag, appliance))

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
    fig.savefig(os.path.join(PLOT_DIRS["curves"], f"{model_tag}_{appliance}_curves.png"), dpi=150)
    plt.close(fig)


def plot_class_performance(appliance: str, model_tag: str):
    """
    Left: confusion matrix normalized per actual class (each row sums to 100%),
    so OFF and ON errors are comparable despite class imbalance.
    Right: precision, recall and F1 of each class; their mean F1 is the Macro-F1.
    """
    metrics = _load_json(_metrics_path(model_tag, appliance))
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    (tn, fp), (fn, tp) = cm

    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([[f"{row_pct[i, j]:.1f}%\n({int(cm[i, j]):,})" for j in range(2)]
                      for i in range(2)])

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    scores = {"OFF": prf(tn, fn, fp), "ON": prf(tp, fp, fn)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    sns.heatmap(row_pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=100,
                xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"], ax=ax1,
                cbar_kws={"label": "% of actual class"})
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Row-normalized confusion matrix")

    x = np.arange(2)
    width = 0.25
    for i, (name, color) in enumerate(zip(["Precision", "Recall", "F1"],
                                          ["tab:blue", "tab:orange", "tab:green"])):
        vals = [scores["OFF"][i], scores["ON"][i]]
        bars = ax2.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        ax2.bar_label(bars, fmt="%.2f", fontsize=8)
    ax2.axhline(metrics["macro_f1"], color="black", linestyle="--", linewidth=1,
                label=f"Macro-F1 = {metrics['macro_f1']:.3f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"OFF (n={int(tn + fp):,})", f"ON (n={int(fn + tp):,})"])
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("Score")
    ax2.set_title("Per-class precision / recall / F1")
    ax2.legend(fontsize=8, loc="upper center", ncol=4)

    fig.suptitle(f"{model_tag.upper()} - {appliance} Per-Class Performance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRS["class_metrics"],
                             f"{model_tag}_{appliance}_class_performance.png"), dpi=150)
    plt.close(fig)


def plot_training_dynamics(appliance: str, model_tag: str):
    """
    Top: generalization gap (val loss - train loss) per epoch.
    Bottom: learning-rate schedule (warmup + cosine) with validation Macro-F1,
    the selected (best) epoch and the early-stopping patience window.
    """
    history = _load_json(_history_path(model_tag, appliance))
    train = np.array(history["train_loss"])
    val = np.array(history["val_loss"])
    f1 = np.array(history["val_f1"])
    lr = np.array(history["lr"])
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmax(f1)) + 1
    stopped_early = len(epochs) < EPOCHS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    gap = val - train
    ax1.bar(epochs, gap, color=np.where(gap > 0, "tab:red", "tab:green"), alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Val loss - Train loss")
    ax1.set_title("Generalization gap (red = val loss above train loss)")

    ax2.plot(epochs, lr, color="tab:purple", marker="o", markersize=3, label="Learning rate")
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel("Epoch")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax3 = ax2.twinx()
    ax3.plot(epochs, f1, color="tab:green", linestyle="--", label="Val Macro-F1")
    ax3.set_ylabel("Val Macro-F1")

    for ax in (ax1, ax2):
        ax.axvline(best, color="tab:blue", linestyle=":", linewidth=1.5)
    ax3.scatter([best], [f1[best - 1]], color="tab:blue", zorder=5,
                label=f"Selected model (epoch {best}, F1={f1[best - 1]:.3f})")
    if best < len(epochs):
        ax2.axvspan(best + 0.5, len(epochs) + 0.5, color="grey", alpha=0.15,
                    label="No improvement" + (" -> early stop" if stopped_early else ""))

    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, fontsize=8, loc="lower center")
    ax2.set_title(f"LR schedule and model selection "
                  f"({len(epochs)}/{EPOCHS} epochs{', early stopped' if stopped_early else ''})")

    fig.suptitle(f"{model_tag.upper()} - {appliance} Training Dynamics")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRS["dynamics"],
                             f"{model_tag}_{appliance}_training_dynamics.png"), dpi=150)
    plt.close(fig)


def plot_model_comparison():
    """Bar chart comparing all models that have metrics for every appliance."""
    tags = [tag for tag in MODEL_TAGS
            if all(os.path.exists(_metrics_path(tag, a)) for a in TARGET_APPLIANCES)]
    if not tags:
        print("[viz] No complete model metrics found to compare.")
        return

    metrics = {(tag, a): _load_json(_metrics_path(tag, a))
               for tag in tags for a in TARGET_APPLIANCES}

    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    minimums = {"accuracy": 0.85, "balanced_accuracy": 0.75, "macro_f1": 0.70}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(TARGET_APPLIANCES))
    width = 0.8 / len(tags)

    for ax, metric in zip(axes, metric_names):
        for i, tag in enumerate(tags):
            vals = [metrics[(tag, a)][metric] for a in TARGET_APPLIANCES]
            offset = (i - (len(tags) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=tag.upper())
        ax.axhline(minimums[metric], color="red", linestyle="--", linewidth=1,
                   label="CCP minimum")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_APPLIANCES, rotation=15)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Model Comparison: " + " vs ".join(t.upper() for t in tags))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIRS["comparison"], "model_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots():
    """Generate every required plot."""
    for tag in MODEL_TAGS:
        for appliance in TARGET_APPLIANCES:
            if not os.path.exists(_metrics_path(tag, appliance)):
                print(f"[viz] Skipping {tag}/{appliance} - metrics not found")
                continue
            plot_confusion_matrix(appliance, tag)
            plot_class_performance(appliance, tag)
            if os.path.exists(_history_path(tag, appliance)):
                plot_training_curves(appliance, tag)
                plot_training_dynamics(appliance, tag)
    plot_model_comparison()
    print("[viz] All plots saved to sub-folders of", os.path.dirname(METRICS_DIR))


if __name__ == "__main__":
    generate_all_plots()
