import gc
import logging
import random
import torch
import transformers
from .sap_data import SapData
from .ps_data import PsData
from .experiment import ExperimentConfig
from .checkpoint import EnsembleCheckpoint

logger = logging.getLogger(__name__)


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
        tag: str,
        seed: int = 1432,
        dataset_name="VCMDATA",
        init_train_factor=2,
    ):
        self.dataset_name = dataset_name
        self.config = config
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

        init_train_size = len(self.train_dataset) // init_train_factor
        pool_data_size = len(self.train_dataset) - init_train_size
        generator = torch.Generator().manual_seed(seed)
        self.run_dataset, self.pool_dataset = torch.utils.data.random_split(
            self.train_dataset, [init_train_size, pool_data_size], generator=generator
        )
        logger.debug("BertExperimentDatasets - data splits: init_train_factor: %d, init_train_size: %d, len(train_dataset): %d, len(run_dataset): %d, len(pool_dataset): %d",
                     init_train_factor,
                     init_train_size,
                     len(self.train_dataset),
                     len(self.run_dataset),
                     len(self.pool_dataset)
                    )

    def update_checkpoint(self, tag):
        self.ckpt.ckpt_tag = tag

    def __update_by_intuition(
        self, size, epi_indices, ale_indices, rank_func, bad_indices, good_indices
    ):
        n = 0
        while epi_indices and n < size:
            logger.debug(f"enter changeset selection loop: len(epi_indices)={len(epi_indices)}, n = {n} < size = {size}")
            candidate_epi = rank_func(epi_indices, key=lambda u: u["epi"])
            candidate_ale = rank_func(ale_indices, key=lambda u: u["ale"])
            if candidate_epi["index"] == candidate_ale["index"]:
                logger.debug(
                    "before rejecting bad index {}, len(epi_indices) = {} len(ale_indices) =  {}, len(bad_indices) = {}".format(
                        candidate_epi["index"],
                        len(epi_indices),
                        len(ale_indices),
                        len(bad_indices),
                    )
                )
                bad_indices.append(candidate_epi["index"])
                epi_indices.remove(candidate_epi)
                ale_indices.remove(candidate_ale)
                logger.debug(
                    "after rejecting bad index {}, len(epi_indices) = {} len(ale_indices) =  {}, len(bad_indices) = {}".format(
                        candidate_epi["index"],
                        len(epi_indices),
                        len(ale_indices),
                        len(bad_indices),
                    )
                )
                logger.debug(
                    "rejected index {}, due to  candidate_epi {} == candidate_ale {}".format(
                        candidate_epi["index"], candidate_epi, candidate_ale
                    )
                )
            else:
                logger.debug(
                    "before accepting good index {}, len(epi_indices) = {} len(ale_indices) =  {}".format(
                        candidate_epi["index"], len(epi_indices), len(ale_indices)
                    )
                )
                good_indices.append(candidate_epi["index"])
                epi_indices.remove(candidate_epi)
                ale_indices[:] = list(
                    filter(lambda u: u["index"] != candidate_epi["index"], ale_indices)
                )[:]
                logger.debug(
                    "after accepting index {}, len(epi_indices) = {} len(ale_indices) =  {}".format(
                        candidate_epi["index"], len(epi_indices), len(ale_indices)
                    )
                )
                n += 1
                logger.debug(
                    "accepted index {}, due to candidate_epi {} != candidate_ale {}".format(
                        candidate_epi["index"], candidate_epi, candidate_ale
                    )
                )
                logger.debug(
                    "len(good_indices) = {}, wanted size = {}, n = {} now".format(
                        len(good_indices), size, n
                    )
                )
        logger.debug("leave __update_by_intuition ...")

    def update_by_intuition(
        self, size, entropy_epistermic, entropy_aleatoric, method, tag
    ):
        logger.debug("enter update_by_intuition ...")

        assert (
            len(self.pool_dataset) == len(entropy_epistermic) == len(entropy_aleatoric)
        )
        logger.debug(
            "len(pool_dataset): {}, len(entropy_epistermic): {}, len(entropy_aleatoric): {}".format(
                len(self.pool_dataset), len(entropy_epistermic), len(entropy_aleatoric)
            )
        )

        self.ckpt.ckpt_tag, self.ckpt.min_total_loss = tag, None
        epi_indices = [{"index": i, "epi": u} for i, u in enumerate(entropy_epistermic)]
        ale_indices = [{"index": i, "ale": u} for i, u in enumerate(entropy_aleatoric)]
        bad_indices, good_indices = [], []
        if method == "ehal_max":  # case 1
            self.__update_by_intuition(
                size, epi_indices, ale_indices, max, bad_indices, good_indices
            )
        elif method == "elah_max":  # case 2
            self.__update_by_intuition(
                size, epi_indices, ale_indices, min, bad_indices, good_indices
            )
        else:
            raise ValueError("unimplemented method {}".format(method))
        if len(good_indices) >= size:
            indices = good_indices
        else:
            logger.warning(
                "there is no more good canndidates: len(epi_indicies) = {}, len(ale_indices) = {}".format(
                    len(epi_indices), len(ale_indices)
                )
            )
            logger.debug(
                "len(good_indices) = {}, len(epi_indicies) = {}, len(ale_indices) = {}".format(
                    len(good_indices), len(epi_indices), len(ale_indices)
                )
            )
            assert len(epi_indices) == len(ale_indices) == 0
            if size - len(good_indices) <= len(bad_indices):
                indices = good_indices + bad_indices[0 : (size - len(good_indices))]
                logger.warning(
                    "not enough good indices, use {} bad indices from {} bad indicies".format(
                        size - len(good_indices), len(bad_indices)
                    )
                )
            else:
                indices = good_indices + bad_indices
                logger.warning(
                    "not enough good indices, use all {} bad indices".format(
                        len(bad_indices)
                    )
                )
        selected = torch.utils.data.Subset(self.pool_dataset, indices)
        logger.debug(
            "selected {} instances using method {}".format(len(selected), method)
        )
        self.run_dataset = torch.utils.data.ConcatDataset([self.run_dataset, selected])
        logger.debug("rundataset is now size {}".format(len(self.run_dataset)))

        indices = [i for i in range(len(self.pool_dataset)) if i not in indices]
        self.pool_dataset = torch.utils.data.Subset(self.pool_dataset, indices)
        logger.debug("pool dataset is now size {}".format(len(self.pool_dataset)))
        logger.info(
            "method: {}, len(run_dataset): {}, len(pool_dataset): {}".format(
                method, len(self.run_dataset), len(self.pool_dataset)
            )
        )
        logger.debug("leave update_by_intuition ...")

    def update_by_random(
        self, size, entropy_epistermic, entropy_aleatoric, method, tag
    ):
        self.ckpt.ckpt_tag, self.ckpt.min_total_loss = tag, None
        uq_list = list(
            zip(range(len(self.pool_dataset)), entropy_epistermic, entropy_aleatoric)
        )
        if method != "random":  # case 1
            raise ValueError("incorrect method {}".format(method))

        # randomly select size samples and remove them from uq list
        logger.debug(
            "sampling size {} from len(uq_list): {}".format(size, len(uq_list))
        )
        indices = [uq[0] for uq in random.sample(uq_list, size)]

        selected = torch.utils.data.Subset(self.pool_dataset, indices)
        self.run_dataset = torch.utils.data.ConcatDataset([self.run_dataset, selected])
        logger.debug("rundataset is now size {}".format(len(self.run_dataset)))

        indices = [i for i in range(len(self.pool_dataset)) if i not in indices]
        self.pool_dataset = torch.utils.data.Subset(self.pool_dataset, indices)
        logger.debug("pool dataset is now size {}".format(len(self.pool_dataset)))
        logger.info(
            "method: {}, len(run_dataset): {}, len(pool_dataset): {}".format(
                method, len(self.run_dataset), len(self.pool_dataset)
            )
        )


    def update_by_score(
        self, size, entropy_epistermic, entropy_aleatoric, method, tag, train=False
    ):
        if self.test_dataset is not None and train:
            self.test_dataset = None
            gc.collect()
        self.ckpt.ckpt_tag, self.ckpt.min_total_loss = tag, None
        uq_list = list(
            zip(range(len(self.pool_dataset)), entropy_epistermic, entropy_aleatoric)
        )
        if method == "ehal":  # case 1
            sorted_uq_list = sorted(uq_list, key=lambda u: u[1] / u[2], reverse=True)
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

    def update(
        self, size, entropy_epistermic, entropy_aleatoric, method, tag, train=False
    ):
        if self.test_dataset is not None and train:
            self.test_dataset = None
            gc.collect()
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

    def _generate_datasets(self):
        if self.dataset_name == "PSDATA" or self.dataset_name == "VCMDATA":
            bert_data = PsData(self.config.data.data_dir)
            logger.info(
                "loading data set {} from {}".format(
                    self.dataset_name, self.config.data.data_dir
                )
            )
        elif self.dataset_name == "SAPDATA":
            bert_data = SapData(self.config.data.data_dir)
            logger.info(
                "loading data set {} from {}".format(
                    self.dataset_name, self.config.data.data_dir
                )
            )
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
        elif self.config.trainer.use_data == "from_scratch":
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
                batch_size=config.trainer.batch_size,
                num_workers=config.trainer.num_dataloader_workers,
                pin_memory=config.trainer.pin_memory,
            )
        logger.info(
            f"len(run_dataset) of len(train_datasetf): {len(datasets.run_dataset)} of {len(datasets.train_dataset)}"
        )
