"""
gru_model_explained.py

transcribed from the user's screenshots, with:
- chinese comments translated to english
- clearer explanations added throughout
- a final big-picture note on why this kind of model is used in quant work

this is a straightforward GRU + MLP forecasting model.
"""

from torch import nn
import torch


class Model(torch.nn.Module):
    """
    simple GRU-based sequence model.

    parameters
    ----------
    num_i : int
        number of input features at each time step.
        example: returns, volume, spread, imbalance, factor values, etc.
    num_h : int
        hidden size of the GRU and the first dense layer width.
    num_o : int
        output dimension.
        for a single forecast target, this is often 1.
    seq_len : int, default=10
        length of the input sequence window.
        only matters when all_steps=True, because then all GRU outputs are flattened.
    dropout : float, default=0.05
        dropout probability used in the MLP head.
    all_steps : bool, default=False
        if False:
            use only the last GRU output as the summary representation.
        if True:
            flatten all GRU outputs across time and feed the whole sequence
            representation into the dense layers.
    """

    def __init__(self, num_i, num_h, num_o, seq_len=10, dropout=0.05, all_steps=False):
        super(Model, self).__init__()

        self.num_h = num_h
        self.all_steps = all_steps

        # GRU layer:
        # input feature dimension = num_i
        # hidden dimension = num_h
        # single-layer GRU
        #
        # batch_first=True means the input shape is:
        # (batch_size, seq_len, num_i)
        self.gru = nn.GRU(
            input_size=num_i,
            hidden_size=num_h,
            num_layers=1,
            batch_first=True
        )

        # optional batch normalization, kept commented because it was commented
        # out in the screenshot source.
        # self.bn1 = nn.BatchNorm1d(num_h)

        # fully connected head
        #
        # two modes:
        # 1) all_steps=False:
        #    use only the final time step hidden state, shape (batch_size, num_h)
        # 2) all_steps=True:
        #    flatten all time steps, shape (batch_size, seq_len * num_h)
        if all_steps:
            self.linear1 = nn.Linear(seq_len * num_h, num_h)
        else:
            self.linear1 = nn.Linear(num_h, num_h)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_h, int(num_h / 2))
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(int(num_h / 4), num_o)

        # dropout probability must be a float
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        forward pass.

        input
        -----
        x : torch.Tensor
            shape = (batch_size, seq_len, num_i)

        returns
        -------
        out : torch.Tensor
            if num_o == 1, the final squeeze(-1) makes the output shape:
            (batch_size,)

            if num_o > 1, the output shape stays:
            (batch_size, num_o)
        """

        # GRU output:
        # gru_out shape = (batch_size, seq_len, num_h)
        #
        # for each time step, the GRU produces a hidden representation.
        # the second returned object is the final hidden state, but this code
        # does not use it directly.
        gru_out, _ = self.gru(x)

        if self.all_steps:
            # keep information from all time steps by flattening the sequence
            # dimension together with the hidden dimension:
            # (batch_size, seq_len, num_h) -> (batch_size, seq_len * num_h)
            x = gru_out.flatten(start_dim=1)
        else:
            # use only the final time step representation:
            # (batch_size, seq_len, num_h) -> (batch_size, num_h)
            x = gru_out[:, -1, :]

        # optional batch normalization
        # x = self.bn1(x)

        # dense block 1
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # dense block 2
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        # dense block 3
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        # final prediction layer
        x = self.linear4(x)

        # if num_o == 1, this removes the trailing singleton dimension:
        # (batch_size, 1) -> (batch_size,)
        out = x.squeeze(-1)
        return out


# ---------------------------------------------------------------------
# big picture: why this is used in a quant model
# ---------------------------------------------------------------------
#
# this model is used when you want to map a short history of market features
# into a prediction target.
#
# typical quant inputs:
# - recent returns
# - rolling volatility
# - volume or turnover
# - order-book imbalance
# - spreads
# - factor values across recent days or bars
# - macro or cross-asset features through time
#
# typical targets:
# - next-period return
# - direction label
# - volatility forecast
# - probability of a move
# - short-horizon signal score
#
# why use a GRU here:
# - it is built for sequential data
# - it can compress the recent path of the input sequence into a hidden state
# - it is usually lighter and easier to train than an LSTM
# - it can capture time dependence better than a plain feedforward model that
#   only sees one snapshot
#
# why the MLP head exists:
# - the GRU extracts a time-series representation
# - the dense layers transform that representation into the final prediction
# - the progressive shrinking of dimensions acts like a funnel from latent
#   representation to target output
#
# what all_steps changes:
# - all_steps=False:
#   assume the final hidden representation is enough
# - all_steps=True:
#   allow the model to use the whole sequence output, not just the last step
#
# what this model does NOT do automatically:
# - it does not prevent leakage
# - it does not handle transaction costs
# - it does not know anything about execution
# - it does not guarantee economic meaning
# - it does not create alpha by itself
#
# in quant, the real edge usually comes from:
# - better features
# - better target construction
# - strict walk-forward validation
# - leakage control
# - sensible portfolio construction and cost modeling
#
# bluntly: this is just a sequence forecasting block. whether it is useful
# depends far more on the data pipeline and evaluation design than on the fact
# that it uses a GRU.
"""

