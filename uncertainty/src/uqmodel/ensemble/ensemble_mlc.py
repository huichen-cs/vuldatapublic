"""
An ensemble of simple MLP classifiers.
"""
import numpy as np
import torch
import torchmetrics
from typing import Sequence

from uqmodel.ensemble.dataloader_utils import get_test_label
from uqmodel.ensemble.stochastic_metrics import softmax_batch
from uqmodel.ensemble.mlc import MultiLayerClassifier


class StochasticEnsembleClassifier(object):
    """
    An ensemble of classifiers.

    """

    def __init__(
        self,
        Classifier,
        log_sigma: bool,
        ensemble_size: int,
        input: int,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation=None,
    ):
        self.size = ensemble_size
        self.model_ensemble = [
            Classifier(
                input, log_sigma, output, neurons, dropouts, activation=activation
            )
            for i in range(ensemble_size)
        ]
        self.log_sigma = log_sigma

    def to(self, device: torch.device):
        self.model_ensemble = [model.to(device) for model in self.model_ensemble]
        return self

    def __getitem__(self, idx: int):
        return self.model_ensemble[idx]

    def __len__(self):
        return len(self.model_ensemble)

    # def select_model(self, test_dataloader, selection_critieria='best_f1', device=None):
    #     scores = np.zeros(self.__len__())
    #     for i in range(self.__len__()):
    #         if selection_critieria == 'best_f1':
    #             test_label_pred = self._predict_class_by_individual(i, test_dataloader, device=device)
    #             test_label_pred_list = list(test_label_pred)
    #             test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).to(device)
    #             test_label_tensor = get_test_label(test_dataloader, device=device)
    #             f1 = torchmetrics.functional.classification.binary_f1_score(test_label_pred_tensor, test_label_tensor)
    #             scores[i] = f1
    #         else:
    #             raise ValueError('unsupported selection_criteria {}'.format(selection_critieria))
    #     idx = np.argmax(scores)
    #     return self.model_ensemble[idx]

    def predict_proba(
        self, test_dataloader: torch.utils.data.DataLoader, n_samples: int, device=None
    ):
        # testing
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                proba_list = []
                for idx in range(self.size):
                    self.model_ensemble[idx].eval()
                    mu, sigma = self.model_ensemble[idx](x)
                    _, proba = softmax_batch(
                        mu, sigma, n_samples, passed_log_sigma=self.log_sigma
                    )
                    proba_list.append(proba)
                    self.model_ensemble[idx].train()
                yield proba_list

    def predict_mean_proba(
        self,
        test_dataloader: torch.utils.data.DataLoader,
        n_samples: int,
        device: torch.device = None,
    ):
        test_proba = self.predict_proba(test_dataloader, n_samples, device)
        for test_batch in test_proba:
            mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
            yield mean_proba

    # def predict(self, test_dataloader, device=None):
    #     test_proba = self.predict_proba(test_dataloader, device)
    #     for test_batch in test_proba:
    #         mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
    #         yield torch.argmax(mean_proba, dim=1)

    @classmethod
    def compute_class_with_conf(self, mean_proba_iterable):
        for mean_proba_batch in mean_proba_iterable:
            proba, labels = torch.max(mean_proba_batch, dim=-1)
            yield proba, labels

    # def _predict_proba_by_individual(self, model_idx, test_dataloader, device=None):
    #     assert 0 <= model_idx < self.__len__()
    #     # testing
    #     with torch.no_grad():
    #         for test_batch in test_dataloader:
    #             x, _ = test_batch
    #             if device is not None:
    #                 x = x.to(device)
    #             self.model_ensemble[model_idx].eval()
    #             logits = self.model_ensemble[model_idx](x)
    #             proba_batch = torch.softmax(logits, dim=1)
    #             self.model_ensemble[model_idx].train()
    #             yield proba_batch

    # def _predict_class_by_individual(self, model_idx, test_dataloader, device=None):
    #     for proba_batch in self._predict_proba_by_individual(model_idx, test_dataloader, device):
    #         yield torch.argmax(proba_batch, dim=1)


class EnsembleClassifier(object):
    """
    An ensemble of classifiers.

    """

    def __init__(
        self,
        ensemble_size: int,
        input: int,
        output: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation=None,
    ):
        self.n_outputs = output
        self.size = ensemble_size
        self.model_ensemble = [
            MultiLayerClassifier(input, output, neurons, dropouts, activation)
            for i in range(ensemble_size)
        ]

    def to(self, device: torch.device):
        self.model_ensemble = [model.to(device) for model in self.model_ensemble]
        return self

    def __getitem__(self, idx):
        return self.model_ensemble[idx]

    def __len__(self):
        return len(self.model_ensemble)

    def predict_logits(self, data_loader: torch.utils.data.DataLoader):
        if next(self.model_ensemble[0].parameters()).get_device() >= 0:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        batch_list = []
        with torch.no_grad():
            for batch in data_loader:
                x, _ = batch
                batch_size, output_size = x.shape[0], self.n_outputs
                batch_logits = torch.zeros(self.size, batch_size, output_size).to(
                    device
                )
                x = x.to(device)
                for model_idx, model in enumerate(self.model_ensemble):
                    model.eval()
                    logits = model(x)
                    batch_logits[model_idx, :, :] = logits
                    model.train()
                batch_list.append(batch_logits.transpose(1, 0))
        return torch.cat(batch_list, dim=0)

    def predict_logits_proba(self, data_loader: torch.utils.data.DataLoader):
        logits = self.predict_logits(data_loader)
        proba = torch.softmax(logits, dim=-1)
        # mean_proba = torch.mean(proba, dim=-2)
        # conf_pred, label_pred = torch.max(mean_proba, dim=-1)
        # label_indices = label_pred.unsqueeze(-1).repeat(1, 5).unsqueeze(-1)
        # proba_pred = proba.gather(-1, label_indices).squeeze(-1)
        # return logits, conf_pred, label_pred, proba_pred
        return logits, proba

    def select_model(self, test_dataloader, selection_critieria="best_f1", device=None):
        if selection_critieria == "random":
            idx = np.random.randint(0, high=self.__len__(), dtype=int)
            return self.model_ensemble[idx]

        scores = np.zeros(self.__len__())
        for i in range(self.__len__()):
            if selection_critieria in ["best_f1", "median_f1"]:
                test_label_pred = self._predict_class_by_individual(
                    i, test_dataloader, device=device
                )
                test_label_pred_list = list(test_label_pred)
                test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).to(
                    device
                )
                test_label_tensor = get_test_label(test_dataloader, device=device)
                f1 = torchmetrics.functional.classification.binary_f1_score(
                    test_label_pred_tensor, test_label_tensor
                )
                scores[i] = f1
            else:
                raise ValueError(
                    "unsupported selection_criteria {}".format(selection_critieria)
                )
        if selection_critieria == "best_f1":
            idx = np.argmax(scores)
        elif selection_critieria == "median_f1":
            idx = scores.tolist().index(
                np.percentile(scores, 50, interpolation="nearest")
            )
        else:
            pass
        return self.model_ensemble[idx]

    def predict_proba(self, test_dataloader, device=None):
        # testing
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                # y = y.to(device)
                proba_list = []
                for model in self.model_ensemble:
                    model.eval()
                    logits = model(x)
                    proba = torch.softmax(logits, dim=1)
                    proba_list.append(proba)
                    model.train()
                yield proba_list

    def predict_mean_proba(self, test_dataloader, device=None):
        test_proba = self.predict_proba(test_dataloader, device)
        for test_batch in test_proba:
            mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
            yield mean_proba

    def predict(self, test_dataloader, device=None):
        test_proba = self.predict_proba(test_dataloader, device)
        for test_batch in test_proba:
            mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
            yield torch.argmax(mean_proba, dim=1)

    @classmethod
    def compute_class_with_conf(self, mean_proba_iterable, device=None):
        for mean_proba_batch in mean_proba_iterable:
            proba, labels = torch.max(mean_proba_batch, dim=1)
            yield proba, labels

    def _predict_proba_by_individual(self, model_idx, test_dataloader, device=None):
        assert 0 <= model_idx < self.__len__()
        # testing
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                self.model_ensemble[model_idx].eval()
                logits = self.model_ensemble[model_idx](x)
                proba_batch = torch.softmax(logits, dim=1)
                self.model_ensemble[model_idx].train()
                yield proba_batch

    def _predict_class_by_individual(self, model_idx, test_dataloader, device=None):
        for proba_batch in self._predict_proba_by_individual(
            model_idx, test_dataloader, device
        ):
            yield torch.argmax(proba_batch, dim=1)
