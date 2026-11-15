"""
LSTM model for NILM (LayerNorm on the last hidden state, stat features
concatenated before the heads). See recurrent_base.py.
"""
import torch.nn as nn
from src.models.recurrent_base import RecurrentNILM


class NILM_LSTM(RecurrentNILM):
    rnn_class = nn.LSTM
