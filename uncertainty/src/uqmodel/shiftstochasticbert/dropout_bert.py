import torch
from uqmodel.shiftstochasticbert.stochastic_bert_mlc import StochasticBertBinaryClassifier

class DropoutBertClassifier(object):
    def __init__(self, model:StochasticBertBinaryClassifier, n_stochastic_passes:int=100):
        self.model = model
        if isinstance(model, StochasticBertBinaryClassifier):
            self.n_outputs = 2
        else:
            raise ValueError('epected StochasticBertBinaryClassifier but found {}'.format(
                type(model)
            ))
        self.n_stochastic_passes = n_stochastic_passes

    def to(self, device:torch.DeviceObjType):
        self.model = self.model.to(device)
        return self

    def predict(self, input_ids, attention_mask, n_stochastic_passes:int=None):
        if n_stochastic_passes is None:
            n_passes = self.n_stochastic_passes
        else:
            n_passes = n_stochastic_passes

        proba_list = []
        with torch.no_grad():
            for _ in range(n_passes):
                self.model.train() # set train model for stochastic samples
                proba = self.model.predict_proba(input_ids, attention_mask)
                proba_list.append(proba)
        dropout_proba = torch.stack(proba_list)
        mean_proba = dropout_proba.mean(dim=0)
        confidence, labels = torch.max(mean_proba, dim=1)
        return dropout_proba, mean_proba, confidence, labels

