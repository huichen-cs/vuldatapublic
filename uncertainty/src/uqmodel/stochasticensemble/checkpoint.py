from __future__ import annotations
import logging
import os
import pickle  # nosec
import torch
from typing import List, Tuple, Union

from .stochastic_mlc import StochasticMultiLayerClassifier
from .ps_data import FeatureDataSet

logger = logging.getLogger(__name__)


class EnsembleCheckpoint(object):
    """Save and load checkpoint."""

    def __init__(
        self,
        ensemble_size: int,
        checkpoint_dirpath: str,
        warmup_epochs: int = 5,
        min_member_losses: Union[List[float], None] = None,
        tag: Union[str, None] = None,
        train: bool = True,
    ):
        self._ckpt_path = checkpoint_dirpath
        self.ensemble_size = ensemble_size
        if not os.path.exists(self._ckpt_path):
            os.makedirs(self._ckpt_path)
        if min_member_losses is None:
            self._min_member_losses = [None] * ensemble_size
            self._init_min_member_losses = [None] * ensemble_size
        else:
            self._min_member_losses = min_member_losses
            self._init_min_member_losses = min_member_losses.copy()
        self._warmup_epochs = warmup_epochs
        self._checkpointed = False
        self._tag = tag
        self._train = train

    @property
    def warmup_epochs(self) -> int:
        return self._warmup_epochs

    @warmup_epochs.setter
    def warmup_epochs(self, epochs) -> None:
        self._warmup_epochs = epochs

    @property
    def min_total_loss(self) -> float:
        return self._min_total_loss

    @min_total_loss.setter
    def min_total_loss(self, loss: float):
        self._min_total_loss = loss

    @property
    def min_member_losses(self) -> List[float]:
        return self._min_member_losses

    @property
    def ckpt_path(self) -> str:
        return self._ckpt_path

    @property
    def ckpt_tag(self) -> Union[str, None]:
        return self._tag

    @ckpt_tag.setter
    def ckpt_tag(self, _tag: Union[str, None]):
        self._tag = _tag

    def ckpt_dataset_path(self, ds_type: str) -> str:
        if not self._train:
            file_path = os.path.join(self._ckpt_path, f"dataset_{ds_type}.pickle")
            return file_path

        if self._tag:
            file_path = os.path.join(
                self._ckpt_path, f"dataset_{ds_type}_{self._tag}.pickle"
            )
        else:
            file_path = os.path.join(self._ckpt_path, f"dataset_{ds_type}.pickle")
        return file_path

    def ckpt_model_path(self, model_index: int) -> str:
        if self._tag:
            file_path = os.path.join(
                self._ckpt_path, f"model_{model_index}_{self._tag}"
            )
        else:
            file_path = os.path.join(self._ckpt_path, f"model_{model_index}")
        return file_path

    def chkt_model_meta_path(self, use_default: bool = False) -> str:
        if self._tag and not use_default:
            en_filepath = os.path.join(self._ckpt_path, "model_en_{}".format(self._tag))
        else:
            en_filepath = os.path.join(self._ckpt_path, "model_en")
        return en_filepath

    def min_loss_updated(self, index: int, epoch: int, loss: float) -> bool:
        logger.debug(
            f"Checkpoint: checking for min loss: {self._min_member_losses} loss: {loss}"
        )
        if epoch >= self._warmup_epochs:
            if self._min_member_losses[index] and loss < self._min_member_losses[index]:
                self._min_member_losses[index] = loss
                return True
            elif not self._min_member_losses[index]:
                self._min_member_losses[index] = loss
                return True
            else:
                return False
        else:
            return False

    def reset(self):
        self._min_member_losses = self._init_min_member_losses.copy()
        return self

    def save_ensemble_meta(self, ensemble_size: int, total_loss: float) -> None:
        en_filepath = self.chkt_model_meta_path()
        en_dict = {"ensemble_size": ensemble_size, "total_loss": total_loss}
        torch.save(en_dict, en_filepath)
        logger.info("saved checkpoint meta at {}".format(en_filepath))

    def save_member_checkpoint(
        self,
        epoch: int,
        index: int,
        model: StochasticMultiLayerClassifier,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criteria: torch.nn.modules.loss._Loss,
        min_member_loss: float,
    ) -> None:
        file_path = self.ckpt_model_path(index)
        torch.save(
            {
                "epoch": epoch,
                "index": index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "criteria_state_dict": criteria.state_dict(),
                "min_member_loss": min_member_loss,
            },
            file_path,
        )
        logger.info(
            "saved checkpoint ensemble member {} at {}".format(index, file_path)
        )

    def load_model_meta(self) -> dict:
        try:
            en_filepath = self.chkt_model_meta_path(use_default=False)
            en_dict = torch.load(en_filepath)
        except FileNotFoundError:
            try:
                en_filepath = self.chkt_model_meta_path(use_default=True)
                en_dict = torch.load(en_filepath)
            except FileNotFoundError as err:
                raise err
        logger.info("load model meta for ensemble at {}".format(en_filepath))
        return en_dict

    def load_member_checkpoint(
        self,
        model_idx: int,
        model: StochasticMultiLayerClassifier,
        optimizer: Union[torch.optim.Adam, torch.optim.AdamW],
        scheduler: torch.optim.lr_scheduler.StepLR,
        criteria: torch.nn.CrossEntropyLoss,
    ) -> Tuple[
        int,
        StochasticMultiLayerClassifier,
        Union[torch.optim.Adam, torch.optim.AdamW],
        torch.optim.lr_scheduler.StepLR,
        torch.nn.CrossEntropyLoss,
        float,
    ]:
        try:
            file_path = self.ckpt_model_path(model_idx)
            member_checkpoint = torch.load(file_path)
        except FileNotFoundError as err:
            raise err
        epoch = member_checkpoint["epoch"]
        model_idx = member_checkpoint["index"]
        model.load_state_dict(member_checkpoint["model_state_dict"])
        optimizer.load_state_dict(member_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(member_checkpoint["scheduler_state_dict"])
        criteria.load_state_dict(member_checkpoint["criteria_state_dict"])
        min_member_loss = member_checkpoint["min_member_loss"]
        logger.info("load model {} for ensemble at {}".format(model_idx, file_path))
        return (epoch, model, optimizer, scheduler, criteria, min_member_loss)

    def save_datasets(
        self,
        train_dataset: FeatureDataSet,
        val_dataset: FeatureDataSet,
        test_dataset: FeatureDataSet,
        ps_columns: List,
    ) -> None:
        for ds_type, ds in zip(
            ["train", "val", "test", "ps_columns"],
            [train_dataset, val_dataset, test_dataset, ps_columns],
        ):
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, "wb") as f:
                pickle.dump(ds, f)

    def load_datasets(
        self,
    ) -> Tuple[FeatureDataSet, FeatureDataSet, FeatureDataSet, List]:
        ds_list = []
        for ds_type in ["train", "val", "test", "ps_columns"]:
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, "rb") as f:
                ds_list.append(pickle.load(f))  # nosec
        return ds_list
