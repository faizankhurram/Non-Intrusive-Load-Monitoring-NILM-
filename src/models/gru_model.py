"""
GRU model for NILM. Same structure as the LSTM model (see recurrent_base.py)
with a GRU cell: two gates instead of three, ~25% fewer recurrent parameters.
"""
import torch.nn as nn
from src.models.recurrent_base import RecurrentNILM


class NILM_GRU(RecurrentNILM):
    rnn_class = nn.GRU
