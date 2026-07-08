"""
rolling_feature_mlp_explained.py

reconstructed from the user's screenshots.

what this file contains:
- the helper operator classes that were shown in the separate `op.py` screenshots
- the main `Model` class that imports and uses those operators
- chinese comments translated into english
- additional explanation for what each block is doing

important honesty:
- the screenshots show several suspicious or unused pieces of code.
  i preserved the visible logic rather than silently rewriting it.
- in particular, the model applies dropout to `x` instead of `out` in several
  places. that means dropout does not affect the MLP activations the way the
  author probably intended.
- several modules are defined but never used in `forward()`.
"""

from torch import nn
import torch


# ---------------------------------------------------------------------
# helper operator layers
# ---------------------------------------------------------------------

class Mean(nn.Module):
    """
    rolling mean operator over the time dimension.

    input shape
    -----------
    X : (batch_size, seq_len, in_units)

    output shape
    ------------
    (batch_size, seq_len - length + 1, in_units)

    logic
    -----
    for each rolling window of size `length`, compute the mean over time
    for every feature, then multiply by a learnable per-feature weight.
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        size = X.shape[1]
        mean_x = torch.mul(
            X[:, :self.length, :].mean(dim=1),
            self.weight.data
        ).unsqueeze(1)

        for i in range(1, size - self.length + 1):
            mean_x = torch.cat([
                mean_x,
                torch.mul(
                    X[:, i:self.length + i, :].mean(dim=1).unsqueeze(1),
                    self.weight.data
                )
            ], 1)

        return mean_x


class Std(nn.Module):
    """
    rolling standard deviation operator over the time dimension.

    same shape behavior as `Mean`, but computes std instead of mean.
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        size = X.shape[1]
        std_x = torch.mul(
            X[:, :self.length, :].std(dim=1),
            self.weight.data
        ).unsqueeze(1)

        for i in range(1, size - self.length + 1):
            std_x = torch.cat([
                std_x,
                torch.mul(
                    X[:, i:self.length + i, :].std(dim=1).unsqueeze(1),
                    self.weight.data
                )
            ], 1)

        return std_x


class Rate(nn.Module):
    """
    rate / ratio operator.

    this computes a simple ratio between the last time step and an earlier
    time step `length` steps back, feature by feature.

    note:
    -----
    the loop for producing rolling rates was commented out in the screenshot,
    so this implementation returns only one time step of rate output:
    shape (batch_size, 1, in_units)

    NaN and inf values are replaced with 0.
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        size = X.shape[1]

        rate_x = torch.mul(
            X[:, size - 1, :] / X[:, size - self.length, :],
            self.weight.data
        ).unsqueeze(1)

        # the original screenshots showed a commented-out rolling loop here

        rate_x = torch.where(torch.isnan(rate_x), torch.full_like(rate_x, 0), rate_x)
        rate_x = torch.where(torch.isinf(rate_x), torch.full_like(rate_x, 0), rate_x)

        return rate_x


class Max(nn.Module):
    """
    rolling max operator over the time dimension.
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        size = X.shape[1]
        max_x = torch.mul(
            torch.max(X[:, :self.length, :], 1).values,
            self.weight.data
        ).unsqueeze(1)

        for i in range(1, size - self.length + 1):
            max_x = torch.cat([
                max_x,
                torch.mul(
                    torch.max(X[:, i:self.length + i, :], 1).values.unsqueeze(1),
                    self.weight.data
                )
            ], 1)

        return max_x


class Min(nn.Module):
    """
    rolling min operator over the time dimension.
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        size = X.shape[1]
        min_x = torch.mul(
            torch.min(X[:, :self.length, :], 1).values,
            self.weight.data
        ).unsqueeze(1)

        for i in range(1, size - self.length + 1):
            min_x = torch.cat([
                min_x,
                torch.mul(
                    torch.min(X[:, i:self.length + i, :], 1).values.unsqueeze(1),
                    self.weight.data
                )
            ], 1)

        return min_x


class Divide(nn.Module):
    """
    pairwise feature ratio operator.

    intended use:
    -------------
    take an input tensor whose last dimension contains two equal-sized feature
    blocks concatenated together, and compute elementwise ratios between the
    first block and the second block.

    example:
    --------
    if the last dimension is [a1, a2, ..., aF, b1, b2, ..., bF],
    the output becomes [a1/b1, a2/b2, ..., aF/bF].

    input shape
    -----------
    X : (batch_size, time_steps, 2 * out)

    output shape
    ------------
    (batch_size, time_steps, out)
    """

    def __init__(self, in_units, out):
        super().__init__()
        self.num = in_units
        self.weight = nn.Parameter(torch.randn(1, out))
        self.length = out

    def forward(self, X):
        x_new = torch.div(X[:, :, 0], X[:, :, self.length])
        x_new = x_new.unsqueeze(2)

        for i in range(1, self.length):
            x_new = torch.cat((
                x_new,
                torch.div(X[:, :, i], X[:, :, self.length + i]).unsqueeze(2)
            ), 2)

        x_new = torch.where(torch.isnan(x_new), torch.full_like(x_new, 0), x_new)
        x_new = torch.where(torch.isinf(x_new), torch.full_like(x_new, 0), x_new)

        return torch.mul(x_new, self.weight.data)


class Mean_res(nn.Module):
    """
    mean residual operator.

    computes:
        last_value - weighted_mean_of_recent_history - bias

    from the screenshot:
    - uses the very last time step
    - compares it against the mean of the last `length` observations before it
    """

    def __init__(self, in_units, length):
        super().__init__()
        self.length = length
        self.weight = nn.Parameter(torch.randn(1, in_units))
        self.bias = nn.Parameter(torch.randn(in_units))

    def forward(self, X):
        return (
            X[:, -1, :].unsqueeze(1)
            - torch.mul(
                X[:, -self.length - 1:-1, :].mean(dim=1).unsqueeze(1),
                self.weight.data
            )
            - self.bias.data
        )


class Corrcoef(nn.Module):
    """
    correlation coefficient matrix operator.

    translated chinese comment:
    'input a tensor matrix with shape (m, n), output its correlation matrix'
    """

    def __init__(self):
        super().__init__()

    def cal_corrcoef(self, x):
        # variance correction factor
        f = (x.shape[0] - 1) / x.shape[0]

        x_reducemean = x - torch.mean(x, dim=0)
        numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]

        var = x.var(dim=0).reshape(x.shape[1], 1)
        denominator = torch.sqrt(torch.matmul(var, var.T)) * f

        corrcoef = numerator / denominator
        return corrcoef

    def forward(self, X):
        return self.cal_corrcoef(X)


# the screenshot showed an RSQ class wrapped inside triple quotes, so it was
# effectively commented out. i am leaving it out of execution for that reason.


# ---------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------

class Model(nn.Module):
    """
    feature-engineered MLP.

    this is not a plain raw-input MLP. before the dense layers, it computes a
    set of handcrafted / learnable rolling summary statistics from the sequence.

    constructor parameters
    ----------------------
    num_i : int
        number of input features at each time step
    num_h : int
        first hidden layer width
    num_o : int
        output dimension
    dropout : float, default=0.05
        dropout probability
    seq_len : int, default=10
        rolling window length used by the helper operators
    """

    def __init__(self, num_i, num_h, num_o, dropout=0.05, seq_len=10):
        super(Model, self).__init__()

        # rolling operators
        self.max1 = Max(num_i, seq_len)
        self.max2 = Max(num_i, seq_len)
        self.min1 = Min(num_i, seq_len)
        self.min2 = Min(num_i, seq_len)
        self.rate1 = Rate(num_i, seq_len)
        self.rate2 = Rate(num_i, seq_len)
        self.mean1 = Mean(num_i, seq_len)
        self.std1 = Std(num_i, seq_len)
        self.mean2 = Mean(num_i, seq_len)
        self.std2 = Std(num_i, seq_len)
        self.mean3 = Mean(num_i, seq_len)

        # feature block derived from pairwise division
        self.divide = Divide(num_i * 2, num_i)

        # correlation operator
        self.corrcoef = Corrcoef()

        # number of operator-derived feature blocks concatenated before linear1
        self.num_op = 5

        # optional batch norm (defined but never used in forward)
        self.bn1 = nn.BatchNorm1d(int(num_h / 2))

        # MLP head
        self.linear1 = torch.nn.Linear(num_i * self.num_op, num_h)
        self.linear2 = torch.nn.Linear(num_h, int(num_h / 2))
        self.linear3 = torch.nn.Linear(int(num_h / 2), int(num_h / 4))
        self.linear4 = torch.nn.Linear(int(num_h / 4), num_o)

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

        # residual mean operator with fixed window length 10
        self.mean_res = Mean_res(num_i, 10)

    def forward(self, x):
        """
        input
        -----
        x : torch.Tensor
            expected shape:
            (batch_size, seq_len, num_i)

        feature construction
        --------------------
        the model computes multiple summary tensors from the sequence:
        - rolling std
        - rolling mean
        - rolling min
        - rolling max
        - residual-to-mean feature

        then it takes the last time step of several of these and concatenates
        them into one feature vector for the MLP head.
        """

        x5 = self.std1(x)
        x6 = self.mean1(x)
        x7 = self.divide(torch.cat([x5, x6], -1))
        x8 = self.min1(x)
        x10 = self.max1(x)
        x14 = self.mean_res(x)

        # start from last-step std and mean
        x1 = torch.cat([x5[:, -1, :], x6[:, -1, :]], -1)

        # this line was commented out in the screenshot
        # x1 = torch.cat([x7[:, -1, :], x1], -1)

        # add last-step min
        x1 = torch.cat([x8[:, -1, :], x1], -1)

        # add last-step max
        x1 = torch.cat([x10[:, -1, :], x1], -1)

        # add mean residual feature
        x1 = torch.cat([x14[:, -1, :], x1], -1)

        # MLP
        out = self.linear1(x1)
        out = self.relu1(out)

        # IMPORTANT:
        # the screenshot applies dropout to `x`, not to `out`.
        # that means dropout does not affect the dense-layer activations here.
        # i preserved that exact logic instead of silently "fixing" it.
        x = self.dropout(x)

        out = self.linear2(out)
        out = self.relu2(out)
        x = self.dropout(x)

        out = self.linear3(out)
        out = self.relu3(out)
        x = self.dropout(x)

        out = self.linear4(out)
        out = out.squeeze(-1)

        return out


# ---------------------------------------------------------------------
# big picture: why this is used in a quant model
# ---------------------------------------------------------------------
#
# this model is trying to do something more specific than a plain MLP.
#
# instead of feeding the raw time-series window directly into the dense layers,
# it first computes hand-designed rolling summary features such as:
# - rolling mean
# - rolling std
# - rolling min
# - rolling max
# - residual from recent mean
# - optional ratio-style derived features
#
# in quant terms, that means the model is mixing:
# 1. feature engineering
# 2. shallow neural-network prediction
#
# why someone would do this:
# - raw sequential data is noisy
# - summary statistics often carry more stable signal than the raw path
# - simple feedforward models can work fine if the right summary features are built
#
# what this is really doing:
# - compressing a short history into engineered state descriptors
# - then using an MLP to map that state into a forecast
#
# examples of what these summary stats may represent:
# - volatility regime (std)
# - local level / average state (mean)
# - recent extremes (min/max)
# - deviation from recent equilibrium (mean_res)
#
# bluntly:
# this is a handcrafted factor-extraction model with a neural head.
# the hard part is not the network. the hard part is whether these rolling
# transformations actually capture predictive structure that survives out of sample.
#
# also, this code has clear issues:
# - many modules are defined but unused
# - dropout is applied to the wrong tensor
# - some parameters like `bias` in helper ops are often defined but not used
# - duplicated operators (max2, min2, rate2, mean2, std2, mean3, corrcoef) are idle
#
# so do not assume this is "good" just because it is complicated.
# it is only useful if the feature design is sound and the implementation bugs are fixed.
"""

