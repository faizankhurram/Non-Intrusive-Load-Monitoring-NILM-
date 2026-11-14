"""
Baseline MLP model for NILM.
Input: flattened window of whole-house power (WINDOW_SIZE,)
Output: ON/OFF probability + predicted appliance power
"""
import torch
import torch.nn as nn
from src.config import WINDOW_SIZE, MLP_HIDDEN, DROPOUT


class NILM_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = WINDOW_SIZE
        for h in MLP_HIDDEN:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Classification head (ON/OFF)
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )
        # Regression head (appliance power)
        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.ReLU(),  # power is non-negative
        )

    def forward(self, x):
        # x: (batch, WINDOW_SIZE)
        features = self.backbone(x)
        cls_out = self.cls_head(features).squeeze(-1)   # (batch,)
        reg_out = self.reg_head(features).squeeze(-1)   # (batch,)
        return cls_out, reg_out
