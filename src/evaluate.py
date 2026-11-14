"""
Evaluate a trained model on the test set.
Reports: Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix,
         MAE, RMSE for power prediction.
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
)
from src.config import BATCH_SIZE, RESULTS_DIR


def evaluate_model(model, test_ds, appliance_name: str, model_tag: str) -> dict:
    """Run evaluation and return metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    all_cls_pred, all_cls_true = [], []
    all_pow_pred, all_pow_true = [], []

    with torch.no_grad():
        for X, y_cls, y_pow in loader:
            X = X.to(device)
            cls_pred, reg_pred = model(X)
            all_cls_pred.append(cls_pred.cpu().numpy())
            all_cls_true.append(y_cls.numpy())
            all_pow_pred.append(reg_pred.cpu().numpy())
            all_pow_true.append(y_pow.numpy())

    cls_pred_prob = np.concatenate(all_cls_pred)
    cls_true = np.concatenate(all_cls_true).astype(int)
    cls_pred = (cls_pred_prob >= 0.5).astype(int)
    pow_pred = np.concatenate(all_pow_pred)
    pow_true = np.concatenate(all_pow_true)

    acc = accuracy_score(cls_true, cls_pred)
    bal_acc = balanced_accuracy_score(cls_true, cls_pred)
    macro_f1 = f1_score(cls_true, cls_pred, average="macro")
    cm = confusion_matrix(cls_true, cls_pred, labels=[0, 1])
    mae = mean_absolute_error(pow_true, pow_pred)
    rmse = float(np.sqrt(mean_squared_error(pow_true, pow_pred)))

    metrics = {
        "appliance": appliance_name,
        "model": model_tag,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "power_mae": round(mae, 2),
        "power_rmse": round(rmse, 2),
        "on_samples": int(cls_true.sum()),
        "off_samples": int((cls_true == 0).sum()),
    }

    print(f"\n[{model_tag}] {appliance_name}:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro-F1:          {metrics['macro_f1']:.4f}")
    print(f"  Power MAE:         {metrics['power_mae']:.2f} W")
    print(f"  Power RMSE:        {metrics['power_rmse']:.2f} W")
    print(f"  Confusion Matrix:  {cm.tolist()}")
    print(f"  ON samples: {metrics['on_samples']:,}  OFF samples: {metrics['off_samples']:,}")

    # Save to JSON
    path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Store predictions for plotting
    pred_path = os.path.join(RESULTS_DIR, f"{model_tag}_{appliance_name}_preds.npz")
    np.savez_compressed(pred_path,
                        cls_true=cls_true, cls_pred=cls_pred,
                        pow_true=pow_true, pow_pred=pow_pred)

    return metrics
