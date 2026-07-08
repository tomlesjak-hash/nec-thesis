"""
cnn-based forecasting/classification blocks often used in quant experiments.

this file is a cleaned transcription of the code shown in the screenshots.
i kept the model logic the same, but i:
1. translated the chinese comments into english,
2. added much heavier explanation,
3. added notes where the original comments looked misleading.

important caveats:
- this is still the original architecture, not a redesigned one.
- some commented-out lines suggest the author experimented with additional weighting,
  pooling, and an extra convolution branch that is not currently used.
- `Model.conv1` hard-codes `in_channels=10`. that means the code assumes the last
  dimension of the original input has size 10 when that branch is used.
- in `Model.forward`, the output of `conv1` is computed but not actually used. only
  the `x2` branch is fed to the dense layers.
- in `MultiScaleCNN`, the original screenshot comments next to branch outputs were
  likely copied from intermediate tensors and do not match the *final* branch output
  shapes. i explain the likely shape flow below.

expected input convention (based on the code):
- input `x` is typically shaped `(batch_size, sequence_length, num_i)`.
- after `x.permute(0, 2, 1)`, the tensor becomes `(batch_size, num_i, sequence_length)`,
  which is the shape required by `nn.Conv1d` in pytorch: `(N, C_in, L)`.

big picture for quant use:
- a 1d cnn is often used when you want the model to detect local temporal patterns:
  short-term momentum, reversal bursts, volatility clusters, volume surges, order-flow
  shapes, or microstructure signatures.
- the single-scale `Model` applies one temporal processing pipeline over the full window.
- the `MultiScaleCNN` version explicitly creates branches over different effective horizons:
  one branch sees the full input window, one sees only the most recent 7 steps, and one
  sees only the most recent 4 steps.
- in quant terms, that is a crude but reasonable way to combine longer-horizon context
  with fresher short-horizon information.
- this kind of model can be used for return prediction, direction classification,
  volatility forecasting, spread prediction, or signal scoring.
- but the model itself is not an edge. if your labels are bad, your features leak future
  information, or your backtest ignores costs and regime shifts, this model will still
  fail no matter how fancy it looks.
"""

import torch
import torch.nn as nn


class Model(torch.nn.Module):
    """
    a simple cnn + mlp model.

    structure:
    1. permute the input so conv1d can operate across the time axis,
    2. run several temporal convolutions,
    3. collapse the final time dimension to a feature vector,
    4. pass that vector through a small mlp.

    parameters
    ----------
    num_i : int
        number of input channels/features seen by conv2/conv3/conv4.
        if the input is `(batch, seq_len, num_i)`, then `num_i` is the feature count.
    num_h : int
        hidden width used by convolutions and dense layers.
    num_o : int
        output dimension.
    dropout : float
        dropout probability.
    """

    def __init__(self, num_i, num_h, num_o, dropout):
        super(Model, self).__init__()

        # optional normalization that was disabled in the original code.
        # self.bn1 = nn.BatchNorm1d(num_i)

        # dense head: maps the convolution features down to the final output.
        self.linear1 = torch.nn.Linear(num_h, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, int(num_h / 2))
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(int(num_h / 4), num_o)
        self.dropout = nn.Dropout(p=dropout)

        # convolution branch 1.
        # note: this branch hard-codes in_channels=10, so it only works when the
        # last axis of the original input has size 10 before the permute pattern used below.
        # because kernel_size=num_i, this acts like a full-width convolution over a length-num_i axis.
        self.conv1 = torch.nn.Conv1d(10, num_h, kernel_size=num_i, stride=1)

        # main temporal convolution stack.
        # input to these layers is expected to be `(batch, num_i, sequence_length)`.
        self.conv2 = torch.nn.Conv1d(num_i, num_h * 2, kernel_size=4, stride=1)
        self.conv3 = torch.nn.Conv1d(num_h * 2, num_h, kernel_size=4, stride=1)
        self.conv4 = torch.nn.Conv1d(num_h, num_h, kernel_size=4, stride=1)

        # optional pooling that was disabled in the original code.
        # self.pool = torch.nn.AvgPool1d(1, stride=1)

        # optional hand-crafted recency weighting that was disabled in the original code.
        # self.weight = torch.tensor([1 / 60 for i in range(1, 121)]).to(torch.device("cuda")).to(torch.float32).unsqueeze(0)

    def forward(self, x):
        """
        forward pass.

        expected input shape
        --------------------
        x : torch.Tensor
            likely `(batch_size, sequence_length, num_i)`.

        returns
        -------
        out : torch.Tensor
            final model output. if `num_o == 1`, this becomes shape `(batch_size,)`
            after `squeeze(-1)`.
        """

        # rearrange so Conv1d sees channels first:
        # from `(batch, seq_len, num_i)` to `(batch, num_i, seq_len)`.
        x2 = x.permute(0, 2, 1)

        # optional manual weighting over one dimension, disabled in the original code.
        # x2 = torch.mul(x2, self.weight)

        # this alternative branch flips dimensions again and feeds conv1.
        # shape becomes `(batch, seq_len, num_i)`.
        x1 = x2.permute(0, 2, 1)

        # original chinese comment: "narrow convolution".
        # more concretely: because kernel_size=num_i, this tends to collapse one axis to length 1
        # when the kernel spans the whole available width.
        x1 = self.conv1(x1)  # often `(bs, num_h, 1)` if dimensions line up as intended.

        # main temporal stack over the full history window.
        x2 = self.conv2(x2)  # e.g. `(bs, num_h * 2, 7)` if the input length was 10.
        x2 = self.conv3(x2)  # e.g. `(bs, num_h, 4)`.
        x2 = self.conv4(x2)  # e.g. `(bs, num_h, 1)`.

        x2 = self.dropout(x2)

        # optional pooling, disabled in the original code.
        # x2 = self.pool(x2)

        # the original author also considered squeezing and concatenating x1.
        # note that x1 is currently computed but not used.
        # x1 = x1.squeeze(-1)  # would give shape `(bs, num_h)`.

        # remove the final singleton time dimension.
        x2 = x2.squeeze(-1)

        # original commented-out fusion:
        # x = torch.cat([x1, x2], 1)
        # current live code only keeps the x2 branch.
        x = x2

        # dense prediction head.
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

        # if the output has a trailing singleton dimension, remove it.
        out = x.squeeze(-1)
        return out


class MultiScaleCNN(torch.nn.Module):
    """
    a multi-branch cnn that extracts features from different effective horizons.

    idea:
    - branch 1: process the full history window,
    - branch 2: process only the last 7 steps,
    - branch 3: process only the last 4 steps,
    - concatenate all branch outputs,
    - pass the merged feature vector through an mlp.

    this is a common pattern when you suspect that information lives on multiple
    time scales: very recent moves, short lookback patterns, and somewhat longer
    local structure.
    """

    def __init__(self, num_i, num_h, num_o, dropout):
        super(MultiScaleCNN, self).__init__()

        # because we concatenate three branch outputs of width `num_h`,
        # the first dense layer takes `3 * num_h` inputs.
        self.linear1 = torch.nn.Linear(3 * num_h, num_h)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, int(num_h / 2))
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(int(num_h / 4), num_o)
        self.dropout = nn.Dropout(p=dropout)

        # branch 1: three conv layers over the full input window.
        self.branch1 = nn.Sequential(
            torch.nn.Conv1d(num_i, num_h * 2, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(num_h * 2, num_h, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(num_h, num_h, kernel_size=4, stride=1),
            torch.nn.ReLU()
        )

        # branch 2: two conv layers, but only on the last 7 observations.
        self.branch2 = nn.Sequential(
            torch.nn.Conv1d(num_i, num_h, kernel_size=4, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(num_h, num_h, kernel_size=4, stride=1),
            torch.nn.ReLU()
        )

        # branch 3: one conv layer on only the last 4 observations.
        self.branch3 = nn.Sequential(
            torch.nn.Conv1d(num_i, num_h, kernel_size=4, stride=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        """
        expected input shape
        --------------------
        x : torch.Tensor
            likely `(batch_size, sequence_length, num_i)`.

        returns
        -------
        out : torch.Tensor
            final prediction tensor.
        """

        # rearrange for Conv1d: `(batch, seq_len, num_i)` -> `(batch, num_i, seq_len)`.
        x2 = x.permute(0, 2, 1)

        # optional manual weighting from the original experimental code.
        # x2 = torch.mul(x2, self.weight)

        # old single-branch experiment kept for reference in comments.
        # x1 = x2.permute(0, 2, 1)
        # x1 = self.conv1(x1)  # often `(bs, num_h, 1)`.

        # branch 1 sees the full window.
        # if the sequence length is 10, the length progression is:
        # 10 -> 7 -> 4 -> 1.
        x_b1 = self.branch1(x2)

        # branch 2 sees only the most recent 7 points.
        # if the input length is exactly 7 here, the progression is:
        # 7 -> 4 -> 1.
        b2_input = x2[:, :, -7:]
        x_b2 = self.branch2(b2_input)

        # branch 3 sees only the most recent 4 points.
        # with kernel size 4, that directly collapses length 4 -> 1.
        b3_input = x2[:, :, -4:]
        x_b3 = self.branch3(b3_input)

        # optional pooling from older experiments.
        # x2 = self.pool(x2)
        # x1 = x1.squeeze(-1)  # would create a `(bs, something)` vector.

        # each branch should now have a singleton time axis that we remove.
        x_b1 = x_b1.squeeze(-1)
        x_b2 = x_b2.squeeze(-1)
        x_b3 = x_b3.squeeze(-1)

        # concatenate features from all time scales.
        x = torch.cat([x_b1, x_b2, x_b3], 1)

        # dense head.
        x = self.linear1(x)
        x = self.relu1(x)
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


# -----------------------------------------------------------------------------
# big-picture quant interpretation
# -----------------------------------------------------------------------------
# why would someone use this in a quant model?
#
# because market data is sequential and often locally structured.
# a cnn is good at learning reusable local filters such as:
# - a short burst in returns,
# - a volatility spike,
# - a sequence of widening spreads,
# - a volume surge after a quiet period,
# - a short order-flow imbalance pattern.
#
# the single-scale `Model` says:
# "let me process the whole lookback window with one temporal pipeline, then map that
# representation to a prediction."
#
# the `MultiScaleCNN` says:
# "i do not trust one horizon alone. i want one branch for the full window, one for the
# recent 7 observations, and one for the most recent 4 observations, then i will fuse them."
#
# in finance, that makes sense when you believe:
# - older observations still matter a bit,
# - but the freshest few observations may carry disproportionate signal,
# - and the relationship is nonlinear enough that a simple linear factor model may miss it.
#
# typical targets in quant could be:
# - next-period return,
# - sign of return,
# - realized volatility,
# - spread change,
# - short-term alpha score,
# - probability of a liquidity event.
#
# what this model does *not* solve for you:
# - data leakage,
# - bad label design,
# - unstable regimes,
# - transaction costs,
# - slippage,
# - position sizing,
# - portfolio construction,
# - risk control.
#
# so the ruthless version is this:
# the architecture may help learn patterns, but it is not the edge.
# the edge, if there is one, comes from better data, better targets, better validation,
# and a realistic trading pipeline.
