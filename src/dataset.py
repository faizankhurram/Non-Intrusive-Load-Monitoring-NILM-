"""
PyTorch Dataset wrapping the windowed numpy arrays.
Applies StandardScaler (fit on training data only).
Optionally appends statistical features to each window.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.windowing import load_windows
from src.config import USE_STAT_FEATURES


def _add_stat_features(X: np.ndarray) -> np.ndarray:
    """Append statistical features to each window: mean, std, max, min, slope."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    mn = X.min(axis=1, keepdims=True)
    # Linear slope (change per timestep)
    slope = ((X[:, -1] - X[:, 0]) / max(X.shape[1] - 1, 1)).reshape(-1, 1)
    return np.hstack([X, mean, std, mx, mn, slope])


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
    Load windows, optionally add stat features, scale, return datasets.
    """
    train_w = load_windows("train")
    val_w = load_windows("val")
    test_w = load_windows("test")

    X_train = train_w["X"]
    X_val = val_w["X"]
    X_test = test_w["X"]

    # Add statistical features before scaling
    if USE_STAT_FEATURES:
        X_train = _add_stat_features(X_train)
        X_val = _add_stat_features(X_val)
        X_test = _add_stat_features(X_test)

    # Fit scaler on training X only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_ds = NILMDataset(X_train, train_w["y_class"], train_w["y_power"], appliance_idx)
    val_ds = NILMDataset(X_val, val_w["y_class"], val_w["y_power"], appliance_idx)
    test_ds = NILMDataset(X_test, test_w["y_class"], test_w["y_power"], appliance_idx)

    return train_ds, val_ds, test_ds, scaler
