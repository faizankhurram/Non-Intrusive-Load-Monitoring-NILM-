"""
Training loop with early stopping, combined classification + regression loss.
Supports checkpointing and resumption.
"""
import os
import json
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE,
    SEED, MODELS_DIR, CLASS_WEIGHT_FACTOR
)

# Global flag for graceful interruption
_STOP_REQUESTED = False


def _handle_signal(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n  [!] Stop requested — finishing current epoch then saving checkpoint...")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(model_tag: str, appliance_name: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}_ckpt.pt")


def save_checkpoint(model, optimizer, epoch, best_val_loss, patience_counter,
                    history, model_tag, appliance_name):
    """Save a full training checkpoint for later resumption."""
    path = _checkpoint_path(model_tag, appliance_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "history": history,
    }, path)
    print(f"  [checkpoint] Saved at epoch {epoch} -> {path}")


def load_checkpoint(model, optimizer, model_tag, appliance_name):
    """Load a checkpoint if one exists. Returns (start_epoch, best_val, patience, history) or None."""
    path = _checkpoint_path(model_tag, appliance_name)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [checkpoint] Resumed from epoch {ckpt['epoch']} <- {path}")
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["patience_counter"],
        ckpt["history"],
    )


def train_model(model, train_ds, val_ds, appliance_name: str, model_tag: str):
    """
    Train a model with checkpoint/resume support.
    Press Ctrl+C once to stop after the current epoch and save a checkpoint.
    Re-running will resume from where it left off.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    # Register signal handler for graceful stop
    old_handler = signal.signal(signal.SIGINT, _handle_signal)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weight for the ON class
    labels = train_ds.y_class.numpy()
    n_on = labels.sum()
    n_off = len(labels) - n_on
    pos_weight = torch.tensor([n_off / max(n_on, 1) * CLASS_WEIGHT_FACTOR],
                              device=device)
    pos_weight = pos_weight.clamp(max=20.0)

    bce_loss_fn = nn.BCELoss(reduction="none")
    mse_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_cls": [], "val_cls": []}
    start_epoch = 1

    save_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}.pt")

    # Try to resume from checkpoint
    ckpt = load_checkpoint(model, optimizer, model_tag, appliance_name)
    if ckpt is not None:
        start_epoch = ckpt[0] + 1
        best_val_loss = ckpt[1]
        patience_counter = ckpt[2]
        history = ckpt[3]
        if start_epoch > EPOCHS:
            print(f"  Already completed all {EPOCHS} epochs.")
            signal.signal(signal.SIGINT, old_handler)
            return model, history

    for epoch in range(start_epoch, EPOCHS + 1):
        # ---- Training ----
        model.train()
        epoch_loss, epoch_cls = 0.0, 0.0
        n_batches = 0
        for X, y_cls, y_pow in train_loader:
            X, y_cls, y_pow = X.to(device), y_cls.to(device), y_pow.to(device)
            cls_pred, reg_pred = model(X)

            # Weighted BCE
            bce = bce_loss_fn(cls_pred, y_cls)
            weights = torch.where(y_cls == 1, pos_weight, torch.ones_like(y_cls))
            cls_loss = (bce * weights).mean()

            reg_loss = mse_loss_fn(reg_pred, y_pow)
            loss = cls_loss + 0.001 * reg_loss  # classification-dominant

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls += cls_loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        avg_train_cls = epoch_cls / n_batches

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_cls_sum, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y_cls, y_pow in val_loader:
                X, y_cls, y_pow = X.to(device), y_cls.to(device), y_pow.to(device)
                cls_pred, reg_pred = model(X)
                bce = bce_loss_fn(cls_pred, y_cls)
                weights = torch.where(y_cls == 1, pos_weight, torch.ones_like(y_cls))
                cls_loss = (bce * weights).mean()
                reg_loss = mse_loss_fn(reg_pred, y_pow)
                loss = cls_loss + 0.001 * reg_loss
                val_loss_sum += loss.item()
                val_cls_sum += cls_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_cls = val_cls_sum / max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_cls"].append(avg_train_cls)
        history["val_cls"].append(avg_val_cls)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Check if stop was requested (Ctrl+C)
        if _STOP_REQUESTED:
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            patience_counter, history, model_tag, appliance_name)
            print(f"  Training paused at epoch {epoch}. Resume with: python run_pipeline.py")
            signal.signal(signal.SIGINT, old_handler)
            raise KeyboardInterrupt("Training paused by user")

    # Training complete — load best model and clean up checkpoint
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Best model saved to {save_path}")

    # Remove checkpoint since training finished
    ckpt_path = _checkpoint_path(model_tag, appliance_name)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Save history
    hist_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    signal.signal(signal.SIGINT, old_handler)
    return model, history
