"""
A simple multiple layer classifier for estimating epistermic uncertainty via ensemble in pytorch.
"""

import torch
from typing import Sequence


class MultiLayerClassifier(torch.nn.Module):
    def __init__(
        self,
        input: int,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation=None,
    ):
        """Initialize a MultiLayerClassifier.

        Initialize a multi-layer neural network, each layer is a linear layer preceded with a dropout layer and
        activated by a RELU activation (by default). The class is organized like a Scikit-Learn classifier.

        Parameters
        ----------
        input :
            size of inputs
        output :
            size of outputs
        neurons :
            size of hidden layers
        dropouts :
            dropout ratio for each dropout layer
        activation :
            activation for each none-dropout layer
        """
        super().__init__()
        assert len(neurons) == len(dropouts)

        layers = []
        for i in range(0, len(neurons)):
            if dropouts[i] is not None:
                layers.append(torch.nn.Dropout(dropouts[i]))
            if i == 0:
                layers.append(torch.nn.Linear(input, neurons[i]))
            elif i == len(neurons) - 1:
                layers.append(torch.nn.Linear(neurons[i - 1], output))
            else:
                layers.append(torch.nn.Linear(neurons[i - 1], neurons[i]))
            if activation is None:
                layers.append(torch.nn.ReLU())
            elif activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "leakyrelu":
                layers.append(torch.nn.LeakyReLU())
            elif isinstance(activation, torch.nn.Module):
                layers.append(activation)
            else:
                raise ValueError("activation is not a torch.nn.Module")

        self.layers = torch.nn.ModuleList(layers)
        self.n_layers = len(self.layers)

    def forward(self, x):
        shape = x.size()
        x = x.view(shape[0], -1)

        for i in range(self.n_layers):
            # print(i, self.layers[i])
            x = self.layers[i](x)
        return x
