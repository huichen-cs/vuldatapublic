import torch
from .stochastic_bert_mlc import StochasticBertBinaryClassifier


class StochasticDropoutBertClassifier(object):
    def __init__(
        self, model: StochasticBertBinaryClassifier, n_stochastic_passes: int = 100
    ):
        self.model = model
        if isinstance(model, StochasticBertBinaryClassifier):
            self.n_outputs = 2
        else:
            raise ValueError(
                "expected StochasticBertBinaryClassifier but found {}".format(
                    type(model)
                )
            )
        self.n_stochastic_passes = n_stochastic_passes

    def to(self, device: torch.DeviceObjType):
        self.model = self.model.to(device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            self.model.eval()
            _, _, proba_pred, confidence_pred, labels_pred = self.model.predict(
                input_ids, attention_mask
            )
        return proba_pred, confidence_pred, labels_pred

    def stochastic_predict(
        self, input_ids, attention_mask, n_stochastic_passes: int = None
    ):
        if n_stochastic_passes is None:
            n_passes = self.n_stochastic_passes
        else:
            n_passes = n_stochastic_passes

        mu_list, sigma_list = [], []
        with torch.no_grad():
            self.model.eval()
            _, _, proba_pred, confidence_pred, labels_pred = self.model.predict(
                input_ids, attention_mask
            )
            for _ in range(n_passes):
                self.model.train()  # set train model for stochastic samples
                logits_mu, logits_sigma = self.model(input_ids, attention_mask)
                mu_list.append(logits_mu)
                sigma_list.append(logits_sigma)
        dropout_mus = torch.stack(mu_list)
        dropout_sigmas = torch.stack(sigma_list)
        return dropout_mus, dropout_sigmas, proba_pred, confidence_pred, labels_pred
