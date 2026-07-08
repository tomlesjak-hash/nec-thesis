"""
two_stage_gru_mlp_explained.py

reconstructed from the user's screenshots.

what this file contains
-----------------------
- `Model_step1`: a GRU + MLP model
- `Model_step2`: a last-timestep MLP model
- `Model_seq`: a wrapper that adds the two predictions together

i translated the chinese comments into english and added explanations.

big idea
--------
this is a simple ensemble-style design:
- one branch uses sequence modeling through a GRU
- one branch ignores the sequence dynamics and only uses the latest snapshot
- the final model adds both outputs

that is a reasonable idea in quant, because sometimes:
- part of the signal is path-dependent
- part of the signal is mostly cross-sectional / snapshot-based
"""

import torch.nn as nn
import torch


class Model_step1(nn.Module):
    """
    GRU + MLP branch.

    parameters
    ----------
    num_i : int
        number of input features at each time step
    num_h : int
        hidden size of the GRU and first dense layer width
    num_o : int
        output dimension
    seq_len : int, default=10
        sequence length used only when `all_steps=True`
    dropout : float, default=0.05
        dropout probability
    all_steps : bool, default=False
        if True, flatten the entire GRU output sequence before the MLP
        if False, use only the last GRU time step
    """

    def __init__(self, num_i, num_h, num_o, seq_len=10, dropout=0.05, all_steps=False):
        super(Model_step1, self).__init__()
        self.num_h = num_h
        self.all_steps = all_steps

        # translated comment:
        # GRU layer (input dimension = num_i, hidden dimension = num_h, single-layer GRU)
        self.gru = nn.GRU(
            input_size=num_i,
            hidden_size=num_h,
            num_layers=1,
            batch_first=True
        )  # replaced LSTM with GRU

        # optional batch normalization
        # self.bn1 = nn.BatchNorm1d(num_h)

        # fully connected block (preserving the original structure)
        if all_steps:
            # translated comment:
            # flattened dimension becomes seq_len * num_h
            self.linear1 = nn.Linear(seq_len * num_h, num_h)
        else:
            self.linear1 = nn.Linear(num_h, num_h)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_h, int(num_h / 2))
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(int(num_h / 4), num_o)

        # translated comment:
        # dropout must be passed in as a float
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        input
        -----
        x : torch.Tensor
            shape = (batch_size, seq_len, num_i)

        returns
        -------
        out : torch.Tensor
            if num_o == 1, shape becomes (batch_size,) after squeeze(-1)
        """

        # translated comment:
        # GRU layer
        # gru_out shape = (batch_size, seq_len, num_h)
        # each time step contains the last-layer hidden state
        gru_out, _ = self.gru(x)

        if self.all_steps:
            x = gru_out.flatten(start_dim=1)
        else:
            # translated comment:
            # take the final time step output as features
            # shape -> (batch_size, num_h)
            x = gru_out[:, -1, :]

        # optional batch normalization
        # x = self.bn1(x)

        # dense block
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.linear4(x)

        # translated comment:
        # output shape: (batch_size,) when num_o == 1
        out = x.squeeze(-1)
        return out


class Model_step2(nn.Module):
    """
    latest-snapshot MLP branch.

    this branch ignores the full sequence dynamics and only uses the last
    time step, then feeds that feature vector through an MLP.
    """

    def __init__(self, num_i, num_h, num_o, dropout):
        super(Model_step2, self).__init__()

        # self.bn1 = nn.BatchNorm1d(num_i)
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, int(num_h / 2))
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(int(num_h / 4), num_o)

        # translated comment:
        # dropout must be passed in as a float
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        input
        -----
        x : torch.Tensor
            shape = (batch_size, seq_len, num_i)

        returns
        -------
        out : torch.Tensor
            if num_o == 1, shape becomes (batch_size,)
        """

        # translated comment:
        # take the last time step and pass it through the MLP
        x = x[:, -1, :]

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.linear4(x)
        out = x.squeeze(-1)
        return out


class Model_seq(nn.Module):
    """
    combine the two branches by simple additive ensembling.

    forward logic
    -------------
    pred1 = step1(x)   # sequence-aware GRU branch
    pred2 = step2(x)   # latest-snapshot MLP branch
    pred  = pred1 + pred2
    """

    def __init__(self, step1: nn.Module, step2: nn.Module):
        super(Model_seq, self).__init__()
        self.step1 = step1
        self.step2 = step2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred1 = self.step1(x)
        pred2 = self.step2(x)
        pred = pred1 + pred2
        return pred


# ---------------------------------------------------------------------
# big picture: why this is used in a quant model
# ---------------------------------------------------------------------
#
# this is a crude two-branch model for financial prediction.
#
# branch 1:
# - GRU + MLP
# - meant to capture path-dependent information from the recent sequence
# - useful when the ordering and dynamics of the last few observations matter
#
# branch 2:
# - plain MLP on the latest time step only
# - meant to capture information already present in the most recent snapshot
# - useful when the current state matters more than the exact path
#
# final combination:
# - simply add the two predictions
#
# why someone would do this:
# - some alpha comes from recent dynamics
# - some alpha comes from the current cross-sectional state
# - adding them is a cheap way to combine both views
#
# what is good about it:
# - easy to understand
# - easy to train
# - sequence branch and snapshot branch are cleanly separated
#
# what is weak about it:
# - the ensemble weight is fixed at 1 + 1
# - there is no learned gating between the two branches
# - there is no calibration of whether step1 or step2 should dominate
# - if both branches learn similar things, you may just be duplicating noise
#
# bluntly:
# the idea is reasonable, but this is a basic ensemble, not a sophisticated one.
# whether it helps depends on whether the sequence branch and snapshot branch are
# actually learning different useful structure.
"""

