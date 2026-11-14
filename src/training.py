"""
Training loop with early stopping, combined classification + regression loss.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE,
    SEED, MODELS_DIR, CLASS_WEIGHT_FACTOR
)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, train_ds, val_ds, appliance_name: str, model_tag: str):
    """
    Train a model and return training history.
    model_tag: 'mlp' or 'lstm'
    """
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

    save_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}.pt")

    for epoch in range(1, EPOCHS + 1):
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

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  Best model saved to {save_path}")

    # Save history
    hist_path = os.path.join(MODELS_DIR, f"{model_tag}_{appliance_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    return model, history
