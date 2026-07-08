"""
mlp_model_explained.py

transcribed from the user's screenshot, with:
- chinese comments translated into english
- clearer explanations added
- a short big-picture note on why this type of model appears in quant work

this is a very simple multilayer perceptron (mlp) for tabular inputs.
"""

from torch import nn
import torch


class Model(torch.nn.Module):
    """
    simple feedforward neural network.

    parameters
    ----------
    num_i : int
        input dimension, i.e. number of input features.
    num_h : int
        hidden layer width for the first hidden layer.
    num_o : int
        output dimension. for a single prediction target, this is often 1.
    dropout : float
        dropout probability used after each hidden layer.

    architecture
    ------------
    input
      -> linear(num_i -> num_h)
      -> relu
      -> dropout
      -> linear(num_h -> num_h/2)
      -> relu
      -> dropout
      -> linear(num_h/2 -> num_h/4)
      -> relu
      -> dropout
      -> linear(num_h/4 -> num_o)
    """

    def __init__(self, num_i, num_h, num_o, dropout):
        super(Model, self).__init__()

        # optional batch normalization on the input features
        # this was present in the screenshot but not actually used in forward()
        self.bn1 = nn.BatchNorm1d(num_i)

        # first hidden block
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()

        # second hidden block
        self.linear2 = torch.nn.Linear(num_h, int(num_h / 2))
        self.relu2 = torch.nn.ReLU()

        # third hidden block
        self.linear3 = torch.nn.Linear(int(num_h / 2), int(num_h / 4))
        self.relu3 = torch.nn.ReLU()

        # final output layer
        self.linear4 = torch.nn.Linear(int(num_h / 4), num_o)

        # translated chinese comment:
        # "dropout must be passed in as a float"
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        forward pass.

        input
        -----
        x : torch.Tensor
            shape typically:
            (batch_size, num_i)

        returns
        -------
        out : torch.Tensor
            if num_o == 1, squeeze(-1) changes shape from
            (batch_size, 1) to (batch_size,)

            if num_o > 1, the output remains
            (batch_size, num_o)
        """

        # first hidden block
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # second hidden block
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        # third hidden block
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        # final prediction layer
        x = self.linear4(x)

        # remove the trailing dimension if the output is scalar per sample
        out = x.squeeze(-1)
        return out


# ---------------------------------------------------------------------
# big picture: why this is used in a quant model
# ---------------------------------------------------------------------
#
# this model is the most basic neural-network baseline for tabular factor data.
#
# typical quant inputs:
# - engineered alpha factors
# - order book statistics
# - volume and turnover features
# - recent return features
# - cross-sectional descriptors
#
# typical outputs:
# - next-period return forecast
# - classification score
# - probability of up/down move
# - volatility or spread estimate
#
# why use this kind of model:
# - it is simple
# - fast to train
# - easy to debug
# - useful as a baseline before trying more complex models
#
# what it can do:
# - learn non-linear combinations of input features
# - capture interactions that a linear model would miss
#
# what it cannot do by itself:
# - understand time order unless you manually encode temporal features
# - prevent leakage
# - handle transaction costs or execution
# - guarantee economic meaning
#
# bluntly:
# this is not a fancy architecture. it is a basic tabular prediction block.
# in many quant projects, that is actually useful, because weak data pipelines
# do not become good just because the model is more complicated.
"""

