import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel

from typing import Any, List, Sequence, Union, Dict

from uqmodel.shiftbert.datashift import DataShift


logger = logging.getLogger(__name__)

class MultiLayerClassifierHead(torch.nn.Module):
    def __init__(self,
                 noiser:DataShift,
                 input:int,
                 output:int=2,
                 neurons:Sequence=[300, 300, 300],
                 dropouts:Sequence=[0.25, 0.25, 0.25],
                 activation:Union[str,
                                  torch.nn.ReLU,
                                  torch.nn.LeakyReLU,
                                  None]=None):
        """Initialize a MLP classifier head.

        Initialize a multi-layer neural network, each layer is a linear
        layer preceded with a dropout layer and activated by an 
        activation. 

        Parameters
        ----------
        noiser : DataShift
            the data quality shifter that adds noise to data
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
        self.noiser = noiser
        # trunk-ignore(bandit/B101)
        assert len(neurons) == len(dropouts)
        
        layers:List[torch.nn.Module] = []
        for i in range(0, len(neurons)):
            if dropouts[i] is not None:
                layers.append(torch.nn.Dropout(dropouts[i]))
            if i == 0:
                layers.append(torch.nn.Linear(input, neurons[i]))
            elif i == len(neurons)-1:
                layers.append(torch.nn.Linear(neurons[i-1], output))
            else:
                layers.append(torch.nn.Linear(neurons[i-1], neurons[i]))
            if activation is None:
                layers.append(torch.nn.ReLU())
            elif isinstance(activation, str) and activation.lower() == 'relu':
                layers.append(torch.nn.ReLU())
            elif isinstance(activation, str) and activation.lower() == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())
            elif (isinstance(activation, torch.nn.ReLU)
                  or isinstance(activation, torch.nn.LeakyReLU)):
                layers.append(activation)
            else:
                raise ValueError('activation is not a torch.nn.Module')

        self.layers = torch.nn.ModuleList(layers)
        self.n_layers = len(self.layers)
        
    def forward(self, x):
        logger.debug('Before shift: x.max() = {}, x.min() = {}'.format(x.max(), x.min()))
        x_n = self.noiser.shift(x)
        logger.debug('After shift: x.max() = {}, x.min() = {}'.format(x_n.max(), x_n.min()))
        shape = x_n.size()
        x_n = x_n.view(shape[0], -1)
        
        for i in range(self.n_layers):
            x_n = self.layers[i](x_n)
        return x_n

class BertBinaryClassifier(nn.Module):
    def __init__(self,
                 noiser:DataShift,
                 output:int=2,
                 neurons:Sequence=[300, 300, 300],
                 dropouts:Sequence=[0.25, 0.25, 0.25],
                 activation:Union[str,
                                  torch.nn.ReLU,
                                  torch.nn.LeakyReLU,
                                  None]=None,
                 cache_dir:str='~/.cache'):
        super().__init__()
        cache_dir = os.path.expanduser(cache_dir)
        self.noiser = noiser
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = MultiLayerClassifierHead(
            self.noiser,
            self.bert.config.hidden_size,
            output=output,
            neurons=neurons,
            dropouts=dropouts,
            activation=activation
        )
        logger.info('BertBinaryClassifier = {}'.format(self))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

    def classifier_state_dict(self):
        return self.classifier.state_dict()

    def load_classifier_state_dict(self, state_dict:Dict[str, Any]):
        self.classifier.load_state_dict(state_dict)

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            proba = torch.nn.functional.softmax(logits, dim=1)
            confidence, labels = torch.max(proba, dim=1)
        return logits, proba, confidence, labels

    def predict_proba(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            proba = torch.nn.functional.softmax(logits, dim=1)
        return proba
            # confidence, labels = torch.max(proba, dim=1)
            # return confidence, labels, proba
