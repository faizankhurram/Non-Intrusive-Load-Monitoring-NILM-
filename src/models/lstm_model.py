"""
Advanced LSTM model for NILM.
Input: sequence of whole-house power (WINDOW_SIZE, 1)
Output: ON/OFF probability + predicted appliance power
"""
import torch
import torch.nn as nn
from src.config import WINDOW_SIZE, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT


class NILM_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.dropout = nn.Dropout(DROPOUT)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch, WINDOW_SIZE) -> reshape to (batch, WINDOW_SIZE, 1)
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)          # (batch, seq, hidden)
        last_hidden = lstm_out[:, -1, :]    # (batch, hidden)
        last_hidden = self.dropout(last_hidden)

        cls_out = self.cls_head(last_hidden).squeeze(-1)
        reg_out = self.reg_head(last_hidden).squeeze(-1)
        return cls_out, reg_out
