"""Shared weight initialization for NILM models."""
import math
import torch.nn as nn


def init_linear_relu(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_recurrent(rnn: nn.Module):
    for name, param in rnn.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            if isinstance(rnn, nn.LSTM):
                hidden = param.shape[0] // 4
                param.data[hidden:2 * hidden].fill_(1.0)


def init_transformer(encoder: nn.Module):
    for m in encoder.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_output_bias(cls_head: nn.Sequential, on_ratio: float):
    p = min(max(on_ratio, 1e-4), 1 - 1e-4)
    final = cls_head[-1]
    nn.init.normal_(final.weight, std=0.01)
    nn.init.constant_(final.bias, math.log(p / (1 - p)))
