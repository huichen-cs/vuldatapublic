import logging
import torch
import transformers
from uqmodel.stochasticbert.sap_data import SapData
from uqmodel.stochasticbert.experiment import ExperimentConfig
from uqmodel.stochasticbert.checkpoint import EnsembleCheckpoint

logger = logging.getLogger(__name__)


class TextClassificationDataset(torch.utils.data.Dataset):
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
        # trunk-ignore(bandit/B101)
        assert len(input_ids) == len(attention_mask) == self.max_len
        return input_ids, attention_mask, labels


class BertExperimentDatasets(object):
    POS_FILENAME = "SAP_full_commits.csv"
    NEG_FILENAME = "SAP_negative_commits_10x.csv"

    def __init__(self, config: ExperimentConfig, tag: str, seed: int = 1432):
        self.config = config
        self.ckpt = EnsembleCheckpoint(
            config.trainer_checkpoint_dir_path,
            warmup_epochs=config.trainer_checkpoint_warmup_epochs,
            tag=tag,
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.build_datasets()

        init_train_size = len(self.train_dataset) // 2
        pool_data_size = len(self.train_dataset) - init_train_size
        generator = torch.Generator().manual_seed(seed)
        self.run_dataset, self.pool_dataset = torch.utils.data.random_split(
            self.train_dataset, [init_train_size, pool_data_size], generator=generator
        )

    def update_checkpoint(self, tag):
        self.ckpt.ckpt_tag = tag

    def update(self, size, entropy_epistermic, entropy_aleatoric, method, tag):
        self.ckpt.ckpt_tag, self.ckpt.min_total_loss = tag, None
        uq_list = list(
            zip(range(len(self.pool_dataset)), entropy_epistermic, entropy_aleatoric)
        )
        if method == "ehal":  # case 1
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[1], u[2]))
        elif method == "elah":  # case 2
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[1], -u[2]))
        elif method == "ehah":  # case 3
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[1], -u[2]))
        elif method == "elal":  # case 4
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[1], u[2]))
        #
        elif method == "aleh":  # case 5
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[2], -u[1]))
        elif method == "ahel":  # case 6
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[2], u[1]))
        elif method == "aheh":  # case 7
            sorted_uq_list = sorted(uq_list, key=lambda u: (-u[2], -u[1]))
        elif method == "alel":  # case 8
            sorted_uq_list = sorted(uq_list, key=lambda u: (u[2], u[1]))
        else:
            raise ValueError("unimplemented method {}".format(method))

        indices = [uq[0] for uq in sorted_uq_list[0:size]]
        selected = torch.utils.data.Subset(self.pool_dataset, indices)
        self.run_dataset = torch.utils.data.ConcatDataset([self.run_dataset, selected])

        indices = [uq[0] for uq in sorted_uq_list[size:]]
        self.pool_dataset = torch.utils.data.Subset(self.pool_dataset, indices)
        logger.info(
            "method: {}, len(run_dataset): {}, len(pool_dataset): {}".format(
                method, len(self.run_dataset), len(self.pool_dataset)
            )
        )

    def build_datasets(self):
        if self.config.trainer_use_model == "use_checkpoint":
            try:
                train_dataset, val_dataset, test_dataset = self.ckpt.load_datasets()
                logger.info(
                    "loaded train/val/test datasets from checkpoint at {}".format(
                        self.config.trainer_checkpoint_dir_path
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
        elif self.config.trainer_use_model == "try_checkpoint":
            try:
                train_dataset, val_dataset, test_dataset = self.ckpt.load_datasets()
                logger.info(
                    "loaded train/val/test datasets from checkpoint at {}".format(
                        self.config.trainer_checkpoint_dir_path
                    )
                )
            except FileNotFoundError:
                logger.info(
                    "unable to load checkpoint, prepare data sets "
                    + "with train/test ratios: {} and validation ratio: {}".format(
                        self.config.train_test_ratios, self.config.val_ratio
                    )
                )
                sap_data = SapData(self.config.data_dir)
                data_splits = sap_data.train_test_val_split(
                    self.config.train_test_ratios[0],
                    self.config.train_test_ratios[1],
                    self.config.val_ratio,
                    self.config.imbalance_ratio,
                )
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "microsoft/codebert-base", cache_dir=self.config.cache_dir
                )
                train_dataset = TextClassificationDataset(
                    data_splits["train"],
                    tokenizer,
                    self.config.max_encoding_len,
                    self.config.num_classes,
                )
                val_dataset = TextClassificationDataset(
                    data_splits["val"],
                    tokenizer,
                    self.config.max_encoding_len,
                    self.config.num_classes,
                )
                test_dataset = TextClassificationDataset(
                    data_splits["test"],
                    tokenizer,
                    self.config.max_encoding_len,
                    self.config.num_classes,
                )
                self.ckpt.save_datasets(train_dataset, val_dataset, test_dataset)
        else:
            raise ValueError(
                "unsupported configuration option {} for config.trainer_use_model".format(
                    self.config.trainer_use_model
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
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        self.pool_dataloader = torch.utils.data.DataLoader(
            datasets.pool_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            datasets.val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
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
