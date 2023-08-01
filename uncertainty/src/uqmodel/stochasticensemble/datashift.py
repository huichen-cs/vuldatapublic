import numpy as np
import torch
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from uqmodel.stochasticensemble.ps_data import FeatureDataSet


class DataShift(ABC):
    @abstractmethod
    def shift(self, data):
        while False:
            yield None

    @classmethod
    def from_dict(self, d: dict):
        if "type" not in d:
            raise ValueError("invalid argument, no shift type is given")
        if d["type"] == "IndependentGaussianNoiseDataShift":
            return IndependentGaussianNoiseDataShift.from_dict(d)
        else:
            raise ValueError("unknow shift type {}".format(d["type"]))

    @abstractmethod
    def __str__(self):
        while False:
            yield None


class IndependentGaussianNoiseDataShift(DataShift):
    def __init__(self, mu: float, sigma: float):
        self.sigma = sigma
        self.mu = mu

    def shift(self, X):
        if isinstance(X, torch.Tensor):
            noise = (self.sigma**0.5) * torch.randn(X.shape) + self.mu
            X = X + noise
        elif isinstance(X, pd.DataFrame) or isinstance(X, np.array):
            noise = np.random.normal(self.mu, self.sigma, X.shape)
            X = X + noise
        else:
            raise ValueError("unsupported data type {}".format(type(X)))
        return X

    def __str__(self):
        return "{{type: IndependentGaussianNoiseDataShift, mu: {}, sigma: {}}}".format(
            self.mu, self.sigma
        )

    @classmethod
    def from_dict(self, d):
        return IndependentGaussianNoiseDataShift(d["mu"], d["sigma"])


class ShiftedFeatureDataSet(FeatureDataSet):
    def __init__(self, dataset: FeatureDataSet, datashift: DataShift):
        super(ShiftedFeatureDataSet, self).__init__(
            dataset.df, dataset.feature_list, dataset.label_name, dataset.transform
        )
        self.datashift = datashift

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.df.iloc[idx][self.feature_list].to_frame().T
        y = self.df.iloc[idx][self.label_name]

        if self.datashift is not None:
            X = self.datashift.shift(X)

        if self.transform is not None:
            X = self.transform(X)

        X = torch.from_numpy(X).float()
        y = torch.tensor(y).long()

        return X, y


class PortionShiftedFeatureDataSet(FeatureDataSet):
    def __init__(self, dataset: FeatureDataSet, datashift: DataShift, portion: float):
        super(PortionShiftedFeatureDataSet, self).__init__(
            dataset.df, dataset.feature_list, dataset.label_name, dataset.transform
        )
        self.datashift = datashift

        n_shifted = np.ceil(len(self.df) * portion / 2).astype(int)
        index_0_list = [
            i for i in range(len(self.df)) if self.df.iloc[i][self.label_name] == 0
        ]
        index_1_list = [
            i for i in range(len(self.df)) if self.df.iloc[i][self.label_name] == 1
        ]

        index_0_samples = np.random.choice(index_0_list, size=n_shifted, replace=False)
        index_1_samples = np.random.choice(index_1_list, size=n_shifted, replace=False)
        self.shifted_indices = set(
            np.concatenate((index_0_samples, index_1_samples), axis=0)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.df.iloc[idx][self.feature_list].to_frame().T
        y = self.df.iloc[idx][self.label_name]

        if self.datashift and idx in self.shifted_indices:
            X = self.datashift.shift(X)

        if self.transform is not None:
            X = self.transform(X)

        X = torch.from_numpy(X).float()
        y = torch.tensor(y).long()

        return X, y
