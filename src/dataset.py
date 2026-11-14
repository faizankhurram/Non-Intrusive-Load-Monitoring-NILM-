"""
PyTorch Dataset wrapping the windowed numpy arrays.
Applies StandardScaler (fit on training data only).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.windowing import load_windows


class NILMDataset(Dataset):
    """Dataset for a single appliance index."""

    def __init__(self, X, y_class, y_power, appliance_idx: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class[:, appliance_idx], dtype=torch.float32)
        self.y_power = torch.tensor(y_power[:, appliance_idx], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_power[idx]


def build_datasets(appliance_idx: int):
    """
    Load windows, fit scaler on train, apply to all splits.
    Returns (train_ds, val_ds, test_ds, scaler).
    """
    train_w = load_windows("train")
    val_w = load_windows("val")
    test_w = load_windows("test")

    # Fit scaler on training X only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_w["X"])
    X_val = scaler.transform(val_w["X"])
    X_test = scaler.transform(test_w["X"])

    train_ds = NILMDataset(X_train, train_w["y_class"], train_w["y_power"], appliance_idx)
    val_ds = NILMDataset(X_val, val_w["y_class"], val_w["y_power"], appliance_idx)
    test_ds = NILMDataset(X_test, test_w["y_class"], test_w["y_power"], appliance_idx)

    return train_ds, val_ds, test_ds, scaler
