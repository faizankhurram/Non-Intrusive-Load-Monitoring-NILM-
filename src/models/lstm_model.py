"""LSTM model for NILM."""
import torch.nn as nn
from src.models.recurrent_base import RecurrentNILM


class NILM_LSTM(RecurrentNILM):
    rnn_class = nn.LSTM
