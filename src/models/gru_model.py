"""GRU model for NILM."""
import torch.nn as nn
from src.models.recurrent_base import RecurrentNILM


class NILM_GRU(RecurrentNILM):
    rnn_class = nn.GRU
