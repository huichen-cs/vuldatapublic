"""
PatchScout Dataset.
"""

import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from abc import ABC, abstractmethod
from typing import Union, Tuple
from .repodata_utils import (
    get_dataset_cve_list,
    get_patchscout_feature_list,
    get_repo_list,
    load_dataset_feature_data,
)

logger = logging.getLogger("ps_data")


class DataTransform(ABC):
    @abstractmethod
    def __call__(self, X):
        pass


class StandardScalerTransform(DataTransform):
    def __init__(self, df, feature_list):
        self.init(df, feature_list)

    def __call__(self, X):
        return self.processor.transform(X)

    def init(self, df, feature_list):
        self.feature_list = feature_list
        self.processor = sklearn.preprocessing.StandardScaler()
        self.processor.fit(df[feature_list])


class FeatureDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_list: list,
        label_name: str,
        transform: DataTransform = None,
    ):
        super(FeatureDataSet, self).__init__()
        self.df = df
        self.feature_list = feature_list
        self.label_name = label_name
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.df.iloc[idx][self.feature_list]
        y = self.df.iloc[idx][self.label_name]

        if self.transform is not None:
            X = self.transform(X.to_frame().T)

        X = torch.from_numpy(X).float()
        y = torch.tensor(y).long()

        return X, y


class PsFeatureDataCollection(object):
    def __init__(self, data_dir):
        self.repo_list = get_repo_list(data_dir)
        logger.info("loaded repo list")
        self.cve_list = get_dataset_cve_list(data_dir)
        if isinstance(self.cve_list, list):
            self.cve_list = np.array(self.cve_list)
        logger.info("loaded CVE list with size {}".format(len(self.cve_list)))
        self.ps_columns = get_patchscout_feature_list()
        logger.info("number of features to be used is {}".format(len(self.ps_columns)))
        self.df = load_dataset_feature_data(data_dir, "Dataset_5000", self.repo_list)
        logger.info("loaded dataset with shape {}".format(self.df.shape))


class PsFeatureExperimentDataBuilder(object):
    def __init__(
        self,
        data_collection: PsFeatureDataCollection,
        train_test_ratios: list = (0.8, 0.2),
        val_ratio: float = 0.2,
        imbalance_ratio: int = 10,
        cve_sample_size: Union[int, float] = 100,
        Transform: DataTransform = None,
        shuffle: bool = True,
        seed: int = None,
    ):
        self._collection = data_collection
        (
            self._train_dataset,
            self._train_train_dataset,
            self._train_val_dataset,
            self._test_dataset,
        ) = self._get_train_val_test_dataset(
            train_test_ratios=train_test_ratios,
            val_ratio=val_ratio,
            imbalance_ratio=imbalance_ratio,
            cve_sample_size=cve_sample_size,
            Transform=Transform,
            shuffle=shuffle,
            seed=seed,
        )

    @property
    def ps_columns(self) -> list:
        return self._collection.ps_columns

    @property
    def cve_list(self) -> tuple:
        return tuple(self._collection.cve_list)

    @property
    def train_dataset(self) -> FeatureDataSet:
        return self._train_dataset

    @property
    def train_train_dataset(self) -> FeatureDataSet:
        return self._train_train_dataset

    @property
    def train_val_dataset(self) -> FeatureDataSet:
        return self._train_val_dataset

    @property
    def test_dataset(self) -> FeatureDataSet:
        return self._test_dataset

    def _get_train_val_test_cves(
        self,
        cve_sample_size: Union[int, float] = 100,
        train_test_ratios: list = (0.8, 0.2),
        val_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = None,
    ):
        if shuffle:
            local_cve_list = list(self.cve_list)
            np.random.default_rng(seed).shuffle(local_cve_list)
        if isinstance(cve_sample_size, int):
            cve_list = local_cve_list[0:cve_sample_size]
        else:
            cve_list = local_cve_list[
                0 : np.ceil(len(local_cve_list) * cve_sample_size).astype(np.int)
            ]

        n_train_test_cves = np.cumsum(
            np.round(len(cve_list) * np.array(train_test_ratios)).astype(np.int)
        )
        train_cves = cve_list[0 : n_train_test_cves[0]]
        n_train_val_cves = np.cumsum(
            np.round(len(train_cves) * np.array([1 - val_ratio, val_ratio])).astype(
                np.int
            )
        )
        train_train_cves = train_cves[0 : n_train_val_cves[0]]
        train_val_cves = train_cves[n_train_val_cves[0] : n_train_val_cves[1]]
        test_cves = cve_list[n_train_test_cves[0] : n_train_test_cves[1]]

        return train_cves, test_cves, train_train_cves, train_val_cves

    def _get_train_val_test_splits(self, cve_splits):
        train_cves, test_cves, train_train_cves, train_val_cves = cve_splits
        train_df = self._collection.df[self._collection.df["cve"].isin(train_cves)]
        train_train_df = self._collection.df[
            self._collection.df["cve"].isin(train_train_cves)
        ]
        train_val_df = self._collection.df[
            self._collection.df["cve"].isin(train_val_cves)
        ]
        test_df = self._collection.df[self._collection.df["cve"].isin(test_cves)]
        return train_df, test_df, train_train_df, train_val_df

    def _get_imbalanced_data(self, df, cve_list, imbalance_ratio=10, seed=None):
        rng = np.random.default_rng(seed)
        df_list = []
        for cve in cve_list:
            df_1 = df[(df["cve"] == cve) & (df["label"] == 1)]
            df_0 = df[(df["cve"] == cve) & (df["label"] == 0)]

            if imbalance_ratio * len(df_1) >= len(df_0):
                df_list += [df_1, df_0]
            else:
                df_0 = df_0.iloc[rng.choice(len(df_0), imbalance_ratio)]
                df_list += [df_1, df_0]
        df = pd.concat(df_list)
        return df

    def _get_imbalanced_train_val_test_data(
        self, cve_splits, imbalance_ratio, seed=None
    ):
        train_cves, test_cves, train_train_cves, train_val_cves = cve_splits
        _, test_df, train_train_df, train_val_df = self._get_train_val_test_splits(
            (train_cves, test_cves, train_train_cves, train_val_cves)
        )
        train_train_df = self._get_imbalanced_data(
            train_train_df, train_train_cves, imbalance_ratio=imbalance_ratio, seed=seed
        )
        train_val_df = self._get_imbalanced_data(
            train_val_df, train_val_cves, imbalance_ratio=imbalance_ratio, seed=seed
        )
        test_df = self._get_imbalanced_data(
            test_df, test_cves, imbalance_ratio=imbalance_ratio, seed=seed
        )
        return train_train_df, train_val_df, test_df

    def _get_train_val_test_dataset(
        self,
        train_test_ratios=(0.8, 0.2),
        val_ratio=0.2,
        imbalance_ratio=10,
        cve_sample_size=100,
        Transform=None,
        shuffle=True,
        seed=None,
    ):
        (
            train_cves,
            test_cves,
            train_train_cves,
            train_val_cves,
        ) = self._get_train_val_test_cves(
            cve_sample_size=cve_sample_size,
            train_test_ratios=train_test_ratios,
            val_ratio=val_ratio,
            shuffle=shuffle,
            seed=seed,
        )
        (
            train_train_df,
            train_val_df,
            test_df,
        ) = self._get_imbalanced_train_val_test_data(
            (train_cves, test_cves, train_train_cves, train_val_cves),
            imbalance_ratio=imbalance_ratio,
            seed=seed,
        )
        train_df = pd.concat([train_train_df, train_val_df])
        if Transform is not None:
            transform = Transform(train_df, self.ps_columns)
        else:
            transform = None
        train_dataset = FeatureDataSet(
            train_df, self.ps_columns, "label", transform=transform
        )
        train_train_dataset = FeatureDataSet(
            train_train_df, self.ps_columns, "label", transform=transform
        )
        train_val_dataset = FeatureDataSet(
            train_val_df, self.ps_columns, "label", transform=transform
        )
        test_dataset = FeatureDataSet(
            test_df, self.ps_columns, "label", transform=transform
        )
        return train_dataset, train_train_dataset, train_val_dataset, test_dataset


def get_dataset_stats(ds: FeatureDataSet) -> dict:
    n_rows = len(ds)
    n_cols = ds[0][0].shape[1]
    X_0 = torch.cat([x for x, y in ds if y == 0])
    X_1 = torch.cat([x for x, y in ds if y == 1])
    n_rows_0 = len(X_0)
    n_rows_1 = len(X_1)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_rows_0": n_rows_0,
        "n_rows_1": n_rows_1,
    }


def get_dataloader_shape(dataloader):
    total_len = 0
    for batch in dataloader:
        X, y = batch
        total_len += len(X)
    logger.info(
        "dataloader.shape = ({}, {}, {})".format(total_len, X.shape[-1], len(y.shape))
    )
    return (total_len, X.shape[-1], len(y.shape))
