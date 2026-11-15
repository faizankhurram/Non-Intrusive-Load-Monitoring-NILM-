"""
Advanced LSTM model for NILM with LayerNorm.
Input: sequence of whole-house power + statistical features
Output: ON/OFF probability + predicted appliance power
"""
import torch
import torch.nn as nn
from src.config import (
    WINDOW_SIZE, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT,
    USE_STAT_FEATURES, N_STAT_FEATURES,
)


class NILM_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = WINDOW_SIZE
        self.n_stat = N_STAT_FEATURES if USE_STAT_FEATURES else 0

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(LSTM_HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

        # Combine LSTM output with stat features
        head_in = LSTM_HIDDEN + self.n_stat

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, WINDOW_SIZE + n_stat)
        seq = x[:, :self.seq_len].unsqueeze(-1)  # (batch, seq_len, 1)
        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Concatenate stat features if present
        if self.n_stat > 0:
            stat_feats = x[:, self.seq_len:]      # (batch, n_stat)
            combined = torch.cat([last_hidden, stat_feats], dim=1)
        else:
            combined = last_hidden

        cls_out = self.cls_head(combined).squeeze(-1)
        reg_out = self.reg_head(combined).squeeze(-1)
        return cls_out, reg_out
