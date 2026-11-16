"""Transformer encoder for NILM."""
import torch
import torch.nn as nn
from src.config import (
    WINDOW_SIZE, USE_STAT_FEATURES, N_STAT_FEATURES,
    TF_D_MODEL, TF_HEADS, TF_LAYERS, TF_FF, TF_DROPOUT,
)
from src.models.weight_init import init_linear_relu, init_transformer


class NILM_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = WINDOW_SIZE
        self.n_stat = N_STAT_FEATURES if USE_STAT_FEATURES else 0

        self.embed = nn.Conv1d(1, TF_D_MODEL, kernel_size=5, padding=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, WINDOW_SIZE, TF_D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=TF_D_MODEL,
            nhead=TF_HEADS,
            dim_feedforward=TF_FF,
            dropout=TF_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=TF_LAYERS, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(TF_D_MODEL)
        self.dropout = nn.Dropout(TF_DROPOUT)

        head_in = 2 * TF_D_MODEL + self.n_stat
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
        init_transformer(self.encoder)
        init_linear_relu(self.embed)
        init_linear_relu(self.cls_head)
        init_linear_relu(self.reg_head)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        seq = x[:, :self.seq_len].unsqueeze(1)
        tokens = self.embed(seq).transpose(1, 2) + self.pos_embed
        h = self.norm(self.encoder(tokens))
        pooled = self.dropout(torch.cat([h[:, -1, :], h.mean(dim=1)], dim=1))

        if self.n_stat > 0:
            combined = torch.cat([pooled, x[:, self.seq_len:]], dim=1)
        else:
            combined = pooled

        cls_out = self.cls_head(combined).squeeze(-1)
        reg_out = self.reg_head(combined).squeeze(-1)
        return cls_out, reg_out
