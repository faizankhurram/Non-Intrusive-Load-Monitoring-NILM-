"""
Baseline MLP model for NILM with BatchNorm.
Input: flattened window + statistical features
Output: ON/OFF logit + predicted (normalized) appliance power
"""
import torch.nn as nn
from src.config import WINDOW_SIZE, MLP_HIDDEN, DROPOUT, USE_STAT_FEATURES, N_STAT_FEATURES
from src.models.weight_init import init_linear_relu


class NILM_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = WINDOW_SIZE + (N_STAT_FEATURES if USE_STAT_FEATURES else 0)

        layers = []
        for h in MLP_HIDDEN:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        init_linear_relu(self)

    def forward(self, x):
        features = self.backbone(x)
        cls_out = self.cls_head(features).squeeze(-1)
        reg_out = self.reg_head(features).squeeze(-1)
        return cls_out, reg_out
