import collections
import logging
import os
import torch
from transformers import AutoModel
from typing import Any, Sequence, Union, Dict
from uqmodel.shiftstochasticbert.datashift import DataShift
from uqmodel.shiftstochasticbert.stochastic_metrics import softmax_batch

logger = logging.getLogger(__name__)


class StochasticMultiLayerClassifierHead(torch.nn.Module):
    def __init__(
        self,
        input: int,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation: Union[str, torch.nn.ReLU, torch.nn.LeakyReLU, None] = None,
    ):
        """Initialize a MLP classifier head.

        Initialize a multi-layer neural network, each layer is a linear
        layer preceded with a dropout layer and activated by an
        activation.

        Parameters
        ----------
        input : int
            the size of inputs
        output: int
            the size of outputs, the default is 2
        neurons : Sequence
            the sizes of hidden layers
        dropouts : Sequence
            the dropout ratios for each dropout layer preceding each
            neurons layer
        activation : Union[str, troch.nn.Module]
            the activation layer follows the last neurons layer.
            The default is torch.nn.ReLU().
        """
        super().__init__()
        assert len(neurons) == len(dropouts)

        #                         +-- mu module -> mu logits
        # input -> shared module  |
        #                         +-- sigma module -> sigma logits
        shared_layer_dict, output_mu_dict, output_sigma_dict = (
            collections.OrderedDict(),
            collections.OrderedDict(),
            collections.OrderedDict(),
        )

        # share module
        # share moduule 1: input layer module
        if dropouts[0] is not None:
            shared_layer_dict["dropout_0"] = torch.nn.Dropout(dropouts[0])
        shared_layer_dict["input"] = torch.nn.Linear(input, neurons[0])
        self._add_activiation_layer(shared_layer_dict, "activation_0", activation)

        # share moduule 1: hidden layer module
        for i in range(1, len(neurons)):
            if dropouts[i] is not None:
                shared_layer_dict["dropout_" + str(i)] = torch.nn.Dropout(dropouts[i])
            shared_layer_dict["hidden_" + str(i)] = torch.nn.Linear(
                neurons[i - 1], neurons[i]
            )
            self._add_activiation_layer(
                shared_layer_dict, "activation_" + str(i), activation
            )
        self.shared_layers = torch.nn.Sequential(shared_layer_dict)

        # output layer module: mu and sigma modules
        # mu module
        if dropouts[len(neurons) - 1] is not None:
            output_mu_dict["dropout_mu_" + str(len(neurons) - 1)] = torch.nn.Dropout(
                dropouts[len(neurons) - 1]
            )
        output_mu_dict["output_mu"] = torch.nn.Linear(neurons[len(neurons) - 1], output)
        self._add_activiation_layer(
            output_mu_dict, "activation_mu_" + str(len(neurons) - 1), activation
        )
        self.mu_layers = torch.nn.Sequential(output_mu_dict)

        # sigma module
        if dropouts[len(neurons) - 1] is not None:
            output_sigma_dict[
                "dropout_sigma_" + str(len(neurons) - 1)
            ] = torch.nn.Dropout(dropouts[len(neurons) - 1])
        output_sigma_dict["output_sigma"] = torch.nn.Linear(
            neurons[len(neurons) - 1], output
        )
        self._add_activiation_layer(
            output_sigma_dict, "activation_sigma_" + str(len(neurons) - 1), "softplus"
        )
        self.sigma_layers = torch.nn.Sequential(output_sigma_dict)

    def forward(self, x):
        """
        Compute mu and log(sigma).
        """
        shape = x.size()
        x = x.view(shape[0], -1)

        x = self.shared_layers(x)
        mu = self.mu_layers(x)
        sigma = self.sigma_layers(x)

        return mu, sigma

    def _add_activiation_layer(
        self,
        shared_layer_dict: collections.OrderedDict,
        name: str,
        activation: Union[str, torch.nn.Module],
    ):
        if activation is None:
            shared_layer_dict[name] = torch.nn.LeakyReLU()
        elif isinstance(activation, str) and activation.lower() == "leakyrelu":
            shared_layer_dict[name] = torch.nn.LeakyReLU()
        elif isinstance(activation, str) and activation.lower() == "relu":
            shared_layer_dict[name] = torch.nn.ReLU()
        elif isinstance(activation, str) and activation.lower() == "softplus":
            shared_layer_dict[name] = torch.nn.Softplus()
        elif isinstance(activation, torch.nn.Softplus):
            shared_layer_dict[name] = torch.nn.Softplus()
        else:
            raise ValueError("activation is not a supported activation module")


class StochasticBertBinaryClassifier(torch.nn.Module):
    def __init__(
        self,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation: Union[str, torch.nn.ReLU, torch.nn.LeakyReLU, None] = None,
        cache_dir: str = "~/.cache",
    ):
        super().__init__()
        cache_dir = os.path.expanduser(cache_dir)
        self.bert = AutoModel.from_pretrained(
            "microsoft/codebert-base", cache_dir=cache_dir
        )
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = StochasticMultiLayerClassifierHead(
            self.bert.config.hidden_size,
            output=output,
            neurons=neurons,
            dropouts=dropouts,
            activation=activation,
        )
        logger.info("StochasticBertBinaryClassifier = {}".format(self))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits_mu, logits_sigma = self.classifier(pooled_output)
        return logits_mu, logits_sigma

    def classifier_state_dict(self):
        return self.classifier.state_dict()

    def load_classifier_state_dict(self, state_dict: Dict[str, Any]):
        self.classifier.load_state_dict(state_dict)

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits_mu, logits_sigma = self.forward(input_ids, attention_mask)
            proba = torch.nn.functional.softmax(logits_mu, dim=1)
            confidence, labels = torch.max(proba, dim=1)
        return logits_mu, logits_sigma, proba, confidence, labels

    def predict_sampling_proba(self, input_ids, attention_mask, n_samples):
        with torch.no_grad():
            logits_mu, logits_sigma = self.forward(input_ids, attention_mask)
            _, proba = softmax_batch(
                logits_mu, logits_sigma, n_samples, passed_log_sigma=False
            )
        return proba
