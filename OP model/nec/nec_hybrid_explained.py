"""
nec_hybrid_explained.py

reconstructed from the user's screenshots.

what this file contains
-----------------------
- an LSTM feature extractor (`C_L`)
- a classifier head (`C_F`)
- a plain MLP regressor (`NEC_MLP`)
- a wrapper model (`NEC`) that routes predictions based on the classifier output

i translated the chinese comment into english and added more explanation.

important honesty
-----------------
this design is not clean.

the `NEC` model uses a 3-class softmax classifier, but in the final routing step
it only checks whether `argmax(prob) == 1`. if yes, it uses the `extreme` model;
otherwise it uses the `normal` model. that means classes 0 and 2 are both treated
the same at the routing stage. that may be intentional, but it is also a possible
design inconsistency.
"""

import torch
import torch.nn as nn
from typing import Tuple


class C_L(nn.Module):
    """
    LSTM sequence encoder.

    parameters
    ----------
    input_dim : int, default=1
        number of input features at each time step.
    hidden_dim : int, default=256
        hidden size of the LSTM.
    layer_dim : int, default=8
        number of stacked LSTM layers.
    dropout : float, default=0.4
        dropout applied between LSTM layers.

    input shape
    -----------
    x : (batch_size, seq_len, input_dim)

    output
    ------
    out : (batch_size, seq_len, hidden_dim)
        hidden representation at every time step from the last LSTM layer.
    hn : (layer_dim, batch_size, hidden_dim)
        final hidden state for each layer.
    cn : (layer_dim, batch_size, hidden_dim)
        final cell state for each layer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        layer_dim: int = 8,
        dropout: float = 0.4,
    ):
        super(C_L, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim

        # stacked LSTM
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.layer_dim,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # initialize hidden and cell state on the same device as the input
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        return out, hn, cn


class C_F(nn.Module):
    """
    classifier head on top of the LSTM sequence output.

    this head:
    1. takes one selected time step from the LSTM output
    2. batch-normalizes it
    3. maps it through a linear layer
    4. applies softmax

    by default it uses the last time step (`id=-1`).

    parameters
    ----------
    hidden_dim : int, default=64
        expected hidden dimension of the LSTM output passed into this head.
    output_dim : int, default=3
        number of classes.
    """

    def __init__(self, hidden_dim: int = 64, output_dim: int = 3):
        super(C_F, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim)
        self.L_out1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, id: int = -1) -> torch.Tensor:
        """
        parameters
        ----------
        x : torch.Tensor
            expected shape: (batch_size, seq_len, hidden_dim)
        id : int, default=-1
            which time step to use. -1 means the last time step.

        returns
        -------
        torch.Tensor
            shape: (batch_size, output_dim)

        translated comment from screenshot:
        x[:, id, :] reduces the tensor to shape [batch_size, hidden_dim].
        """
        linear_out = self.L_out1(self.bn1(x[:, id, :]))
        linear_out = self.softmax(linear_out)
        return linear_out


# the screenshot also showed an older, commented-out alternative for C_L.
# it was not active code, so i keep it here as a comment only.
#
# class C_L(nn.Module):
#     def __init__(
#         self,
#         input_dim: int = 1,
#         hidden_dim: int = 256,
#         layer_dim: int = 8,
#         dropout: float = 0.4,
#     ):
#         super(C_L, self).__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.dropout(self.relu(self.linear(x)))
#         return out


class NEC_MLP(nn.Module):
    """
    plain feedforward regressor used inside NEC.

    architecture
    ------------
    num_i -> num_h -> num_h/2 -> num_h/4 -> num_o

    dropout + relu is applied after each hidden layer.
    """

    def __init__(
        self,
        num_i: int = 1,
        num_h: int = 256,
        num_o: int = 1,
        dropout: float = 0.05,
    ):
        super(NEC_MLP, self).__init__()
        self.linear1 = nn.Linear(num_i, num_h)
        self.linear2 = nn.Linear(num_h, num_h // 2)
        self.linear3 = nn.Linear(num_h // 2, num_h // 4)
        self.linear4 = nn.Linear(num_h // 4, num_o)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input shape
        -----------
        x : (batch_size, num_i)

        returns
        -------
        torch.Tensor
            if num_o == 1, output shape becomes (batch_size,)
            after squeeze(-1).
        """
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        out = self.linear4(x).squeeze(-1)
        return out


class NEC(nn.Module):
    """
    hybrid routing model.

    components
    ----------
    cl : nn.Module
        sequence encoder, expected to be something like `C_L`
    cf : nn.Module
        classifier head, expected to be something like `C_F`
    extreme : nn.Module
        regressor used when the classifier predicts the 'extreme' regime/class
    normal : nn.Module
        regressor used otherwise

    forward logic
    -------------
    1. encode the sequence with the LSTM
    2. classify the encoded sequence
    3. extract last-step raw tabular features from `x`
    4. run both regressors on those tabular features
    5. choose one prediction based on the classifier output
    """

    def __init__(
        self,
        cl: nn.Module,
        cf: nn.Module,
        extreme: nn.Module,
        normal: nn.Module,
    ):
        super(NEC, self).__init__()
        self.cl = cl
        self.cf = cf
        self.extreme = extreme
        self.normal = normal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        expected input shape
        --------------------
        x : (batch_size, seq_len, input_dim)

        note on feature slicing
        -----------------------
        the screenshot uses `x[:, -1, :-1]` for both regressors.
        that means:
        - take the last time step only
        - drop the final feature column

        so the last feature is apparently reserved for something not meant to be
        passed into the regressors, or the code author simply excluded it by design.
        """
        out, hn, cn = self.cl(x)
        prob = self.cf(out)

        e_out = self.extreme(x[:, -1, :-1])
        n_out = self.normal(x[:, -1, :-1])

        # routing rule from screenshot:
        # if predicted class == 1, use extreme model output
        # otherwise use normal model output
        pred = torch.where(torch.argmax(prob, dim=1) == 1, e_out, n_out)

        return pred


# ---------------------------------------------------------------------
# big picture: why this is used in a quant model
# ---------------------------------------------------------------------
#
# this is a mixture-of-experts style idea, but implemented in a fairly rough way.
#
# the intuition is:
# - market behavior may not be homogeneous
# - a single regressor may do badly if "normal" periods and "extreme" periods
#   behave differently
# - so first classify the regime, then use a specialized predictor
#
# in quant terms, that means:
# - `C_L` tries to summarize the recent sequence history with an LSTM
# - `C_F` tries to detect what kind of regime or state the current sample belongs to
# - `extreme` and `normal` are two separate prediction models
# - the final output is routed through one of them
#
# why someone would build this:
# - return dynamics may differ in calm vs. stressed periods
# - volatility spikes may need a different model than ordinary market noise
# - sequence information may help classify the current state better than a static snapshot
#
# what is questionable here:
# - the classifier is 3-class, but the routing is effectively binary:
#   class 1 -> extreme, everything else -> normal
# - the classifier and regressors are not tightly coupled by a proper probabilistic mixture
# - hard argmax routing throws away uncertainty information
# - there is no obvious calibration or regime validation in the code itself
#
# bluntly:
# the concept is reasonable, but the implementation is crude.
# it may still work empirically, but you should not assume it is elegant or optimal.
# in a serious quant pipeline, you would want to test whether:
# - the classifier actually separates regimes well
# - the extreme model materially outperforms a single unified model
# - the routing logic improves out-of-sample ic / pnl after costs
"""

