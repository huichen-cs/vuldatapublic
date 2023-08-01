"""
A simple multiple layer classifier for disentangling aleatoric and epistermic uncertainties.
"""

import torch
import collections
from typing import Sequence, Union


class StochasticMultiLayerClassifier(torch.nn.Module):
    def __init__(
        self,
        input: int,
        output_log_sigma: bool,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        *,
        normalization_layers=False,
        activation=None,
    ):
        """Initialize a MultiLayerClassifier.

        Initialize a multi-layer neural network, each layer is a linear layer preceded with a dropout layer and
        activated by a RELU activation (by default). The class is organized like a Scikit-Learn classifier.

        The classifier is a stochastic. Conceptually, the network outputs $N_c$ logits ($z$) where $N_c$ is the
        number of classes. The predicted probabilty of the classes are P(y|x) = softmax(z)$. To model undertainty
        of the prediction, we consider z ~ N(mu, sigma^2). For this, we make the model a two-headed network,
        in particular, it has two output heads, one for the mean and the other log standard deviation. To make
        the model outputs the log standard deviation a pratical consideration because the neural network outputs
        is in (-infty, infty). Alternative, we can also the soft plus activation instead.

        Parameters
        ----------
        input :
            size of inputs
        output_log_sigma : bool
            are sigma outpus log(sigma) or sigma?
        output :
            size of outputs, the number of classes, N_c
        neurons :
            size of hidden layers
        dropouts :
            dropout ratio for each dropout layer
        normalization_layers : bool
            add normalization layer or not
        activation :
            activation, the default is Leaky RELU.
        """
        super().__init__()
        # trunk-ignore(bandit/B101)
        assert len(neurons) == len(dropouts)

        self._output_log_sigma = output_log_sigma

        shared_layer_dict, output_mu_dict, output_sigma_dict = (
            collections.OrderedDict(),
            collections.OrderedDict(),
            collections.OrderedDict(),
        )

        # input module
        if dropouts[0] is not None:
            shared_layer_dict["dropout_0"] = torch.nn.Dropout(dropouts[0])
        shared_layer_dict["input"] = torch.nn.Linear(input, neurons[0])
        if normalization_layers:
            shared_layer_dict["batch_norm_input"] = torch.nn.BatchNorm1d(neurons[0])
        self._add_activiation_layer(shared_layer_dict, "activation_0", activation)

        # hidden layer module
        for i in range(1, len(neurons)):
            if dropouts[i] is not None:
                shared_layer_dict["dropout_" + str(i)] = torch.nn.Dropout(dropouts[i])
            shared_layer_dict["hidden_" + str(i)] = torch.nn.Linear(
                neurons[i - 1], neurons[i]
            )
            if normalization_layers:
                shared_layer_dict["batch_norm_" + str(i)] = torch.nn.BatchNorm1d(
                    neurons[i]
                )
            self._add_activiation_layer(
                shared_layer_dict, "activation_" + str(i), activation
            )

        # output layer module
        if dropouts[len(neurons) - 1] is not None:
            output_mu_dict["dropout_mu_" + str(len(neurons) - 1)] = torch.nn.Dropout(
                dropouts[len(neurons) - 1]
            )
            output_sigma_dict[
                "dropout_sigma_" + str(len(neurons) - 1)
            ] = torch.nn.Dropout(dropouts[len(neurons) - 1])
        output_mu_dict["output_mu"] = torch.nn.Linear(neurons[len(neurons) - 1], output)
        # output_mu_dict['batch_norm_mu'] = torch.nn.BatchNorm1d(output)
        self._add_activiation_layer(
            output_mu_dict, "activation_mu_" + str(len(neurons) - 1), activation
        )

        output_sigma_dict["output_sigma"] = torch.nn.Linear(
            neurons[len(neurons) - 1], output
        )
        # output_sigma_dict['batch_norm_sigma'] = torch.nn.BatchNorm1d(output)
        if self._output_log_sigma:
            self._add_activiation_layer(
                output_sigma_dict,
                "activation_sigma_" + str(len(neurons) - 1),
                activation,
            )
        else:
            self._add_activiation_layer(
                output_sigma_dict,
                "activation_sigma_" + str(len(neurons) - 1),
                "softplus",
            )

        self.shared_layers = torch.nn.Sequential(shared_layer_dict)
        self.mu_layers = torch.nn.Sequential(output_mu_dict)
        self.sigma_layers = torch.nn.Sequential(output_sigma_dict)

    def forward(self, x):
        """
        Compute mu and log(sigma).
        """
        # shape = x.size()
        # x = x.view(shape[0], -1)
        x = x.squeeze()

        x = self.shared_layers(x)
        mu = self.mu_layers(x)
        sigma = self.sigma_layers(x)

        # return mu and log(sigma):
        #   the neural network outputs is $(-\infty, \infty)$.
        #   since $\sigma$, the standard deviation must be in $[0, \infty)$,
        #   we interpret the output as $\log(\sigma)$. To restore $\sigma$,
        #   we $\sigma = e^{\log \sigma}$. Since both $f(x) = \log(x)$ and
        #   $f(x) = e^x$ are both monotonically increaseing in $[0, \infty)$,
        #   there is a one-on-one mapping between the two.
        return mu, sigma

    # def predict(self, x):
    #     x = self.forward(x)
    #     p = torch.softmax(x)
    #     confidence, label = torch.max(p, dim=0)
    #     return confidence, label

    def _add_activiation_layer(
        self,
        shared_layer_dict: collections.OrderedDict,
        name: str,
        activation: Union[str, torch.nn.Module],
    ):
        if activation is None:
            shared_layer_dict[name] = torch.nn.LeakyReLU()
        elif activation == "leakyrelu":
            shared_layer_dict[name] = torch.nn.LeakyReLU()
        elif activation == "softplus":
            shared_layer_dict[name] = torch.nn.Softplus()
        elif isinstance(activation, torch.nn.Module):
            shared_layer_dict[name] = activation
        else:
            raise ValueError("activation is not a torch.nn.Module")

    @property
    def output_log_sigma(self):
        return self._output_log_sigma

    # @classmethod
    # @property
    # def is_log_sigma_model(self):
    #     return self.output_log_sigma()
