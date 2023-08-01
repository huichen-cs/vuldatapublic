import logging
import pandas as pd
import torch
import transformers
from abc import ABC, abstractmethod
from typing import Union, Tuple
from uqmodel.shiftstochasticbert.ps_data import PsData
from uqmodel.shiftstochasticbert.sap_data import SapData
from uqmodel.shiftstochasticbert.experiment import ExperimentConfig
from uqmodel.shiftstochasticbert.checkpoint import EnsembleCheckpoint

logger = logging.getLogger(__name__)


class DataTransform(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


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


class TextClassificationDataset(torch.utils.data.TensorDataset):
    def __init__(self, data, tokenizer, max_len, num_classes):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        message = self.data.iloc[index]["commit_message"]
        patch = self.data.iloc[index]["commit_patch"]
        label = self.data.iloc[index]["label"]

        encoding = self.tokenizer.encode_plus(
            message,
            patch,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        labels = torch.tensor(label)
        assert len(input_ids) == len(attention_mask) == self.max_len
        return input_ids, attention_mask, labels


class BertExperimentDatasets(object):
    def __init__(
        self,
        config: ExperimentConfig,
        tag: Union[str, None],
        seed: int = 1432,
        active_learn: bool = False,
        dataset_name: str = "PSDATA",
    ):
        self.config = config
        self.dataset_name = dataset_name
        self.ckpt = EnsembleCheckpoint(
            config.model.ensemble_size,
            config.trainer.checkpoint.dir_path,
            warmup_epochs=config.trainer.checkpoint.warmup_epochs,
            tag=tag,
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.build_datasets()

        if active_learn:
            init_train_size = len(self.train_dataset) // 2
            pool_data_size = len(self.train_dataset) - init_train_size
            generator = torch.Generator().manual_seed(seed)
            self.run_dataset, self.pool_dataset = torch.utils.data.random_split(
                self.train_dataset,
                [init_train_size, pool_data_size],
                generator=generator,
            )

    def _generate_datasets(self):
        if self.dataset_name == "PSDATA" or self.dataset_name == "VCMDATA":
            bert_data = PsData(self.config.data.data_dir)
            logger.info("loading data set {}".format(self.dataset_name))
        elif self.dataset_name == "SAPDATA":
            bert_data = SapData(self.config.data.data_dir)
            logger.info("loading data set {}".format(self.dataset_name))
        else:
            raise ValueError("unsupported dataset {}".format(self.dataset_name))
        data_splits = bert_data.train_test_val_split(
            self.config.data.train_test_ratios[0],
            self.config.data.train_test_ratios[1],
            self.config.data.val_ratio,
            self.config.data.imbalance_ratio,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/codebert-base", cache_dir=self.config.cache_dir
        )
        train_dataset = TextClassificationDataset(
            data_splits["train"],
            tokenizer,
            self.config.model.max_encoding_len,
            self.config.model.num_classes,
        )
        val_dataset = TextClassificationDataset(
            data_splits["val"],
            tokenizer,
            self.config.model.max_encoding_len,
            self.config.model.num_classes,
        )
        test_dataset = TextClassificationDataset(
            data_splits["test"],
            tokenizer,
            self.config.model.max_encoding_len,
            self.config.model.num_classes,
        )
        return train_dataset, val_dataset, test_dataset

    def build_datasets(self):
        if self.config.trainer.use_data == "use_checkpoint":
            try:
                train_dataset, val_dataset, test_dataset = self.ckpt.load_datasets()
                logger.info(
                    "loaded train/val/test datasets from checkpoint at {}".format(
                        self.config.trainer.checkpoint.dir_path
                    )
                )
            except FileNotFoundError as err:
                logger.info(
                    "unable to load checkpoint, prepare data sets "
                    + "with train/test ratios: {} and validation ratio: {}".format(
                        self.config.train_test_ratios, self.config.val_ratio
                    )
                )
                raise err
        elif self.config.trainer.use_data == "try_checkpoint":
            try:
                train_dataset, val_dataset, test_dataset = self.ckpt.load_datasets()
                logger.info(
                    "loaded train/val/test datasets from checkpoint at {}".format(
                        self.config.trainer.checkpoint.dir_path
                    )
                )
            except FileNotFoundError:
                logger.info(
                    "unable to load checkpoint, prepare data sets "
                    + "with train/test ratios: {} and validation ratio: {}".format(
                        self.config.data.train_test_ratios, self.config.data.val_ratio
                    )
                )
                train_dataset, val_dataset, test_dataset = self._generate_datasets()
                self.ckpt.save_datasets(train_dataset, val_dataset, test_dataset)
        elif self.config.trainer_user_data == "from_scratch":
            train_dataset, val_dataset, test_dataset = self._generate_datasets()
            self.ckpt.save_datasets(train_dataset, val_dataset, test_dataset)
        else:
            raise ValueError(
                "unsupported configuration option {} for config.trainer.use_data".format(
                    self.config.trainer.use_model
                )
            )
        return train_dataset, val_dataset, test_dataset


class BertExperimentDataLoaders(object):
    def __init__(self, config, datasets, train=True):
        self.config = config
        self.datasets = datasets

        self.train_dataloader = None  # prevent logic error

        self.run_dataloader = torch.utils.data.DataLoader(
            datasets.run_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.trainer.num_dataloader_workers,
            pin_memory=config.trainer.pin_memory,
        )
        self.pool_dataloader = torch.utils.data.DataLoader(
            datasets.pool_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.trainer.num_dataloader_workers,
            pin_memory=config.trainer.pin_memory,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            datasets.val_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=config.trainer.num_dataloader_workers,
            pin_memory=config.trainer.pin_memory,
        )
        if train:  # prevent logic error
            self.test_dataloader = None
        else:
            self.test_dataloader = torch.utils.data.DataLoader(
                datasets.test_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )
        logger.info(
            f"len(run_dataset) of len(train_datasetf): {len(datasets.run_dataset)} of {len(datasets.train_dataset)}"
        )
