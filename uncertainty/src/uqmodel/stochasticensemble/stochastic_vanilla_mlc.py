import torch
from .stochastic_mlc import StochasticMultiLayerClassifier
from .stochastic_metrics import softmax_batch


class StochasticVanillaClassifier(object):
    def __init__(
        self, model: StochasticMultiLayerClassifier, n_stochastic_passes: int = 100
    ):
        self.model = model
        if isinstance(model, StochasticMultiLayerClassifier):
            self.n_outputs = 2
        else:
            raise ValueError(
                "expected StochasticMultiLayerClassifier but found {}".format(
                    type(model)
                )
            )
        self.n_stochastic_passes = n_stochastic_passes

    def to(self, device: torch.DeviceObjType):
        self.model = self.model.to(device)
        return self

    def __call__(self, inputs):
        return self.model(inputs)

    def predict(self, inputs):
        with torch.no_grad():
            self.model.eval()
            _, _, proba_pred, confidence_pred, labels_pred = self.model.predict(inputs)
        return proba_pred, confidence_pred, labels_pred

    def stochastic_predict(self, inputs, n_stochastic_passes: int = None):
        if n_stochastic_passes is None:
            n_passes = self.n_stochastic_passes
        else:
            n_passes = n_stochastic_passes

        mu_list, sigma_list = [], []
        with torch.no_grad():
            self.model.eval()
            _, _, proba_pred, confidence_pred, labels_pred = self.model.predict(inputs)
            for _ in range(n_passes):
                self.model.train()  # set train model for stochastic samples
                logits_mu, logits_sigma = self.model(inputs)
                mu_list.append(logits_mu)
                sigma_list.append(logits_sigma)
        dropout_mus = torch.stack(mu_list)
        dropout_sigmas = torch.stack(sigma_list)
        return dropout_mus, dropout_sigmas, proba_pred, confidence_pred, labels_pred

    def generate_sampling_proba(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        n_aleatoric_samples: int,
        device=None,
    ):
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                self.model.eval()  # turn on train for sampling proba/stochastic passes
                mu, sigma = self.model(x)
                _, proba = softmax_batch(
                    mu,
                    sigma,
                    n_aleatoric_samples,
                    passed_log_sigma=self.model.output_log_sigma,
                )
                self.model.train()  # turn on train for sampling proba/stochastic passes
                yield [proba]

    def generate_mean_proba(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        n_samples: int,
        device: torch.device = None,
    ):
        proba = self.generate_sampling_proba(test_dataloader, n_samples, device)
        for test_batch in proba:
            mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
            yield mean_proba

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @classmethod
    def compute_class_with_conf(self, mean_proba_iterable):
        for mean_proba_batch in mean_proba_iterable:
            proba, labels = torch.max(mean_proba_batch, dim=-1)
            yield proba, labels
