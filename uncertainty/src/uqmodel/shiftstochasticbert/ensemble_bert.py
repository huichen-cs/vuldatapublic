from __future__ import annotations
import os
import torch
import collections
from typing import Iterator, List, Optional, Sequence, Union
from .stochastic_bert_mlc import (
    StochasticBertBinaryClassifier,
)
from .datashift import DataShift
from .stochastic_metrics import softmax_batch


class StochasticEnsembleBertClassifier(collections.abc.Iterator):
    """An ensemble of bert classifiers."""

    model_ensemble: List[Optional[StochasticBertBinaryClassifier]]

    def __init__(
        self,
        noiser: DataShift,
        ensemble_size: int,
        num_classes: int = 2,
        neurons: Sequence = [300, 300, 300],
        dropouts: Sequence = [0.25, 0.25, 0.25],
        activation: Union[str, torch.nn.ReLU, torch.nn.LeakyReLU, None] = None,
        cache_dir: str = "~/.cache",
    ):
        cache_dir = os.path.expanduser(cache_dir)
        self.size = ensemble_size
        self.model_ensemble = [
            StochasticBertBinaryClassifier(
                noiser,
                output=num_classes,
                neurons=neurons,
                dropouts=dropouts,
                activation=activation,
                cache_dir=cache_dir,
            )
            for _ in range(ensemble_size)
        ]

    def __getitem__(self, idx: int) -> Union[StochasticBertBinaryClassifier, None]:
        return self.model_ensemble[idx]

    def __len__(self) -> int:
        return self.size

    def __setitem__(self, idx: int, model: StochasticBertBinaryClassifier):
        self.model_ensemble[idx] = model

    def __iter__(self) -> Iterator[StochasticBertBinaryClassifier]:
        self._iter_index = 0
        return self

    def __next__(self) -> Union[StochasticBertBinaryClassifier, None]:
        if self._iter_index < self.__len__():
            model = self.model_ensemble[self._iter_index]
            self._iter_index += 1
            return model
        else:
            raise StopIteration

    def to(
        self, device: Union[torch.device, str, None]
    ) -> StochasticBertBinaryClassifier:
        for i in range(self.size):
            model = self.model_ensemble[i]
            if model:
                self.model_ensemble[i] = model.to(device)
        return self

    def train(self):
        for model in self.model_ensemble:
            model.train()

    def eval(self):
        for model in self.model_ensemble:
            model.eval()

    def predict(self, input_ids, attention_mask):
        proba_list = []
        for model in self.model_ensemble:
            proba = model.predict_proba(input_ids, attention_mask)
            proba_list.append(proba)
        ensemble_proba = torch.stack(proba_list)
        mean_proba = ensemble_proba.mean(dim=0)
        confidence, labels = torch.max(mean_proba, dim=1)
        return ensemble_proba, mean_proba, confidence, labels

    def predict_proba(
        self, test_dataloader: torch.utils.data.DataLoader, n_samples: int, device=None
    ):
        # testing
        with torch.no_grad():
            for test_batch in test_dataloader:
                input_ids, attention_mask, _ = test_batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                proba_list = []

                for idx in range(self.size):
                    self.model_ensemble[idx].to(device)
                    self.model_ensemble[idx].eval()
                    mu, sigma = self.model_ensemble[idx](input_ids, attention_mask)
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
