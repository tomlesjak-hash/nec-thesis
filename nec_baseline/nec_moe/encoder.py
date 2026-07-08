"""GRU sequence encoder (ledger defect 5 fix).

Small and shallow by default (1 layer, hidden 32) per the Module 7 parameter-
count argument: GRU = 3n(n+d+1) parameters per layer vs the prototype's 8-layer
n=256 LSTM (~4M). Deliverable is the final hidden state of the last layer,
``h_T`` — the learned summary statistic the gate routes on.

``bidirectional`` is deliberately not an option: the encoder-only usage makes a
backward pass a leakage question this baseline does not open.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .config import DataConfig, EncoderConfig
from .utils import assert_shape

__all__ = ["GRUEncoder"]


class GRUEncoder(nn.Module):
    """``x_seq (B, T, d_seq) -> h_T (B, hidden_dim)``."""

    def __init__(self, data_cfg: DataConfig, enc_cfg: EncoderConfig) -> None:
        super().__init__()
        self.d_seq = data_cfg.d_seq
        self.seq_len = data_cfg.seq_len
        self.hidden_dim = enc_cfg.hidden_dim
        self.gru = nn.GRU(
            input_size=data_cfg.d_seq,
            hidden_size=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            dropout=enc_cfg.dropout if enc_cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        # No manual h0: PyTorch zero-initializes, removing the prototype's
        # device/shape allocation fuss.

    def forward(self, x_seq: Tensor) -> Tensor:
        assert_shape(x_seq, (None, self.seq_len, self.d_seq), "x_seq")
        _, h_n = self.gru(x_seq)  # h_n: (num_layers, B, hidden)
        h_t = h_n[-1]  # last layer's final hidden state
        assert_shape(h_t, (x_seq.shape[0], self.hidden_dim), "h_T")
        return h_t
