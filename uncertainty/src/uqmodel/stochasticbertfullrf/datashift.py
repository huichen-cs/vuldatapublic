import numpy as np
import torch
import pandas as pd
from abc import ABC, abstractmethod


class DataShift(ABC):
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
        pass

    @abstractmethod
    def shift(self, X):
        pass


class IndependentGaussianNoiseDataShift(DataShift):
    def __init__(self, mu: float, sigma: float):
        self.sigma = sigma
        self.mu = mu

    def shift(self, X):
        if isinstance(X, torch.Tensor):
            noise = (self.sigma**0.5) * torch.randn(X.shape) + self.mu
        elif isinstance(X, pd.DataFrame) or isinstance(X, np.array):
            noise = np.random.normal(self.mu, self.sigma, X.shape)
        else:
            raise ValueError("unsupported data type {}".format(type(X)))
        if X.get_device() >= 0:
            X += noise.cuda()
        else:
            X += noise
        return X

    def __str__(self):
        return "{{type: IndependentGaussianNoiseDataShift, mu: {}, sigma: {}}}".format(
            self.mu, self.sigma
        )

    @classmethod
    def from_dict(self, d):
        return IndependentGaussianNoiseDataShift(d["mu"], d["sigma"])
