"""
Shared architecture for the recurrent NILM models (LSTM and GRU).
Input: sequence of whole-house power + statistical features
Output: ON/OFF logit + predicted (normalized) appliance power
"""
import torch
import torch.nn as nn
from src.config import (
    WINDOW_SIZE, RNN_HIDDEN, RNN_LAYERS, DROPOUT,
    USE_STAT_FEATURES, N_STAT_FEATURES,
)
from src.models.weight_init import init_linear_relu, init_recurrent


class RecurrentNILM(nn.Module):
    rnn_class = None

    def __init__(self):
        super().__init__()
        self.seq_len = WINDOW_SIZE
        self.n_stat = N_STAT_FEATURES if USE_STAT_FEATURES else 0

        self.rnn = self.rnn_class(
            input_size=1,
            hidden_size=RNN_HIDDEN,
            num_layers=RNN_LAYERS,
            batch_first=True,
            dropout=DROPOUT if RNN_LAYERS > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(RNN_HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

        head_in = RNN_HIDDEN + self.n_stat
        self.cls_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        init_linear_relu(self.cls_head)
        init_linear_relu(self.reg_head)
        init_recurrent(self.rnn)

    def forward(self, x):
        # x: (batch, WINDOW_SIZE + n_stat); the label is at the last timestep
        seq = x[:, :self.seq_len].unsqueeze(-1)
        out, _ = self.rnn(seq)
        last_hidden = self.dropout(self.layer_norm(out[:, -1, :]))

        if self.n_stat > 0:
            combined = torch.cat([last_hidden, x[:, self.seq_len:]], dim=1)
        else:
            combined = last_hidden

        cls_out = self.cls_head(combined).squeeze(-1)
        reg_out = self.reg_head(combined).squeeze(-1)
        return cls_out, reg_out
