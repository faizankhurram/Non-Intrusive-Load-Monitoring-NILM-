"""
Training loop with:
  - Focal loss (on logits) for handling class imbalance
  - Normalized power regression as an auxiliary loss
  - AdamW, cosine annealing LR scheduler with warmup
  - Model selection and early stopping on validation Macro-F1
  - Checkpoint/resume support
"""
import os
import json
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
from src.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, MIN_DELTA, WEIGHT_DECAY,
    SEED, MODELS_DIR, CLASS_WEIGHT_FACTOR, MAX_POS_WEIGHT, REG_LOSS_WEIGHT,
    WARMUP_EPOCHS,
)
from src.models.weight_init import init_output_bias

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


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification, computed from logits."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, pos_weight=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target == 1, prob, 1 - prob)
        focal_weight = (1 - pt) ** self.gamma

        if pos_weight is not None:
            class_weight = torch.where(target == 1, pos_weight, torch.ones_like(target))
            focal_weight = focal_weight * class_weight

        loss = self.alpha * focal_weight * bce
        return loss.mean()


def _checkpoint_path(model_tag: str, appliance_name: str) -> str:
    return os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}_ckpt.pt")


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_f1,
                    patience_counter, history, model_tag, appliance_name):
    """Save a full training checkpoint for later resumption."""
    path = _checkpoint_path(model_tag, appliance_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_f1": best_val_f1,
        "patience_counter": patience_counter,
        "history": history,
    }, path)
    print(f"  [checkpoint] Saved at epoch {epoch} -> {path}")


def load_checkpoint(model, optimizer, scheduler, model_tag, appliance_name):
    """Load a checkpoint if one exists."""
    path = _checkpoint_path(model_tag, appliance_name)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  [checkpoint] Resumed from epoch {ckpt['epoch']} <- {path}")
    return (
        ckpt["epoch"],
        ckpt["best_val_f1"],
        ckpt["patience_counter"],
        ckpt["history"],
    )


def train_model(model, train_ds, val_ds, appliance_name: str, model_tag: str):
    """
    Train a model with focal loss, LR scheduling, and checkpoint/resume.
    Press Ctrl+C once to stop after the current epoch and save a checkpoint.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    old_handler = signal.signal(signal.SIGINT, _handle_signal)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    # Compute class weight — use sqrt of ratio to prevent collapse
    labels = train_ds.y_class.numpy()
    n_on = labels.sum()
    n_off = len(labels) - n_on
    raw_ratio = n_off / max(n_on, 1)
    pos_weight_val = min(np.sqrt(raw_ratio) * CLASS_WEIGHT_FACTOR, MAX_POS_WEIGHT)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"  Class balance: ON={int(n_on):,} OFF={int(n_off):,} "
          f"ratio={raw_ratio:.2f} pos_weight={pos_weight_val:.2f}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    init_output_bias(model.cls_head, float(n_on / len(labels)))
    model = model.to(device)

    focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
    mse_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )

    best_val_f1 = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_cls": [], "val_cls": [],
               "val_f1": [], "lr": []}
    start_epoch = 1

    save_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}.pt")

    ckpt = load_checkpoint(model, optimizer, scheduler, model_tag, appliance_name)
    if ckpt is not None:
        start_epoch = ckpt[0] + 1
        best_val_f1 = ckpt[1]
        patience_counter = ckpt[2]
        history = ckpt[3]
        if start_epoch > EPOCHS:
            print(f"  Already completed all {EPOCHS} epochs.")
            signal.signal(signal.SIGINT, old_handler)
            return model, history

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Training ----
        model.train()
        epoch_loss, epoch_cls = 0.0, 0.0
        n_batches = 0
        for X, y_cls, y_pow in train_loader:
            X, y_cls, y_pow = X.to(device), y_cls.to(device), y_pow.to(device)
            cls_logit, reg_pred = model(X)

            cls_loss = focal_loss_fn(cls_logit, y_cls, pos_weight)
            reg_loss = mse_loss_fn(reg_pred, y_pow)
            loss = cls_loss + REG_LOSS_WEIGHT * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls += cls_loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        avg_train_cls = epoch_cls / n_batches

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_cls_sum, val_batches = 0.0, 0.0, 0
        val_pred, val_true = [], []
        with torch.no_grad():
            for X, y_cls, y_pow in val_loader:
                X, y_cls, y_pow = X.to(device), y_cls.to(device), y_pow.to(device)
                cls_logit, reg_pred = model(X)
                cls_loss = focal_loss_fn(cls_logit, y_cls, pos_weight)
                reg_loss = mse_loss_fn(reg_pred, y_pow)
                loss = cls_loss + REG_LOSS_WEIGHT * reg_loss
                val_loss_sum += loss.item()
                val_cls_sum += cls_loss.item()
                val_batches += 1
                val_pred.append((cls_logit >= 0).cpu().numpy())
                val_true.append(y_cls.cpu().numpy())

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_cls = val_cls_sum / max(val_batches, 1)
        val_f1 = f1_score(np.concatenate(val_true).astype(int),
                          np.concatenate(val_pred).astype(int),
                          average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_cls"].append(avg_train_cls)
        history["val_cls"].append(avg_val_cls)
        history["val_f1"].append(float(val_f1))
        history["lr"].append(current_lr)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
              f"val_F1={val_f1:.4f}  lr={current_lr:.6f}")

        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_F1={best_val_f1:.4f})")
                break

        if _STOP_REQUESTED:
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_f1,
                            patience_counter, history, model_tag, appliance_name)
            print(f"  Training paused at epoch {epoch}. Resume with: python run_pipeline.py")
            signal.signal(signal.SIGINT, old_handler)
            raise KeyboardInterrupt("Training paused by user")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Best model saved to {save_path}")

    ckpt_path = _checkpoint_path(model_tag, appliance_name)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    hist_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    signal.signal(signal.SIGINT, old_handler)
    return model, history
