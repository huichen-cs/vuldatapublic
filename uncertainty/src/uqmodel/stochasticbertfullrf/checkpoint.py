import logging
import os
import pickle  # nosec
import torch
from typing import Tuple

logger = logging.getLogger("Bert_EnsembleCheckpoint")


class EnsembleCheckpoint(object):
    """
    Save checkpoint.

    Reference:
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """

    def __init__(
        self,
        checkpoint_dir_path,
        warmup_epochs=5,
        min_loss=None,
        min_member_loss=None,
        tag=None,
        train=True,
    ):
        self._ckpt_path = checkpoint_dir_path
        if not os.path.exists(self._ckpt_path):
            os.makedirs(self._ckpt_path)
        self._min_total_loss = min_loss
        self._min_member_loss = min_member_loss
        self._warmup_epochs = warmup_epochs
        self._checkpointed = False
        self._tag = tag
        self._train = train

    @property
    def warmup_epochs(self):
        return self._warmup_epochs

    @warmup_epochs.setter
    def warmup_epochs(self, epochs):
        self._warmup_epochs = epochs

    @property
    def min_total_loss(self):
        return self._min_total_loss

    @property
    def min_member_loss(self):
        return self._min_member_loss

    @min_total_loss.setter
    def min_total_loss(self, loss):
        self._min_total_loss = loss

    @property
    def ckpt_path(self):
        return self._ckpt_path

    @property
    def ckpt_tag(self):
        return self._tag

    @ckpt_tag.setter
    def ckpt_tag(self, _tag):
        self._tag = _tag

    def ckpt_dataset_path(self, ds_type):
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

    def ckpt_model_path(self, model_index):
        if self._tag:
            file_path = os.path.join(
                self._ckpt_path, f"model_{model_index}_{self._tag}"
            )
        else:
            file_path = os.path.join(self._ckpt_path, f"model_{model_index}")
        return file_path

    def chkt_model_meta_path(self, use_default=False):
        if self._tag and not use_default:
            en_filepath = os.path.join(self._ckpt_path, "model_en_{}".format(self._tag))
        else:
            en_filepath = os.path.join(self._ckpt_path, "model_en")
        return en_filepath

    # def __call__(self, epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss):
    #     logger.debug(f'Checkpoint: checking for min loss: {self._min_total_loss} loss: {total_loss}')
    #     if epoch < self._warmup_epochs:
    #         if  self._min_total_loss and total_loss < self._min_total_loss:
    #             self._min_total_loss = total_loss
    #         elif not self._min_total_loss:
    #             self._min_total_loss = total_loss
    #         return
    #     if (self._min_total_loss and total_loss < self._min_total_loss) or (not self._checkpointed):
    #         self.save_checkpoint(epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss)
    #         self._min_total_loss = total_loss
    #         self._checkpointed = True
    #     if not self._min_total_loss:
    #         self._min_total_loss = total_loss

    # def save_checkpoint(self, epoch, model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble, total_loss):
    #     self.save_ensemble_meta(epoch, len(model_ensemble), total_loss, self._min_total_loss)
    #     for index,(model,optimizer,scheduler,criteria,loss) in enumerate(
    #             zip(model_ensemble, optimizer_ensemble, scheduler_ensemble, criteria_ensemble, loss_ensemble)):
    #         self.save_member_checkpoint(index, model, optimizer, scheduler, criteria, loss)
    #     logger.info(f'saved checkpoint for model {index} to {self._ckpt_path}, loss {self._min_total_loss} -> {total_loss}')

    def min_loss_updated(self, epoch, loss):
        logger.debug(
            f"Checkpoint: checking for min loss: {self._min_member_loss} loss: {loss}"
        )
        if epoch < self._warmup_epochs:
            if self._min_member_loss and loss < self._min_member_loss:
                self._min_member_loss = loss
                return True
            elif not self._min_member_loss:
                self._min_member_loss = loss
                return False
        else:
            if loss < self._min_member_loss:
                self._min_member_loss = loss
                return True
            else:
                return False

    def save_ensemble_meta(self, epoch, ensemble_size, total_loss, min_total_loss):
        en_filepath = self.chkt_model_meta_path()
        en_dict = {
            "epoch": epoch,
            "ensemble_size": ensemble_size,
            "total_loss": total_loss,
            "min_total_loss": min_total_loss,
        }
        torch.save(en_dict, en_filepath)
        logger.info("saved checkpoint meta at {}".format(en_filepath))

    def save_member_checkpoint(
        self, index, model, optimizer, scheduler, criteria, min_member_loss, loss
    ):
        file_path = self.ckpt_model_path(index)
        torch.save(
            {
                "index": index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "criteria_state_dict": criteria.state_dict(),
                "min_member_loss": min_member_loss,
                "loss": loss,
            },
            file_path,
        )
        logger.info(
            "saved checkpoint ensemble member {} at {}".format(index, file_path)
        )

    def load_model_meta(self):
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

    def load_member_checkpoint(self, model_idx, model, optimizer, scheduler, criteria):
        try:
            file_path = self.ckpt_model_path(model_idx)
            member_checkpoint = torch.load(file_path)
        except FileNotFoundError as err:
            raise err
        model.load_state_dict(member_checkpoint["model_state_dict"])
        optimizer.load_state_dict(member_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(member_checkpoint["scheduler_state_dict"])
        criteria.load_state_dict(member_checkpoint["criteria_state_dict"])
        loss = member_checkpoint["loss"]
        min_member_loss = member_checkpoint["min_member_loss"]
        logger.info("load model {} for ensemble at {}".format(model_idx, file_path))
        return model, optimizer, scheduler, criteria, loss, min_member_loss

    def load_checkpoint(
        self,
        model_ensemble,
        optimizer_ensemble,
        scheduler_ensemble,
        criteria_ensemble,
        loss_ensemble,
    ):
        en_dict = self.load_model_meta()
        epoch = en_dict["epoch"]
        total_loss = en_dict["total_loss"]
        ensemble_size = en_dict["ensemble_size"]
        min_total_loss = en_dict["min_total_loss"]
        for i in range(ensemble_size):
            try:
                file_path = self.ckpt_model_path(i)
                member_checkpoint = torch.load(file_path)
            except FileNotFoundError as err:
                raise err
            model_ensemble[i].load_state_dict(member_checkpoint["model_state_dict"])
            optimizer_ensemble[i].load_state_dict(
                member_checkpoint["optimizer_state_dict"]
            )
            scheduler_ensemble[i].load_state_dict(
                member_checkpoint["scheduler_state_dict"]
            )
            criteria_ensemble[i].load_state_dict(
                member_checkpoint["criteria_state_dict"]
            )
            loss_ensemble[i] = member_checkpoint["loss"]
            logger.info("load model {} for ensemble at {}".format(i, file_path))
        return (
            epoch,
            model_ensemble,
            optimizer_ensemble,
            scheduler_ensemble,
            criteria_ensemble,
            loss_ensemble,
            min_total_loss,
            total_loss,
        )

    def save_datasets(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
    ) -> None:
        for ds_type, ds in zip(
            ["train", "val", "test"], [train_dataset, val_dataset, test_dataset]
        ):
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, "wb") as f:
                pickle.dump(ds, f)

    def load_datasets(
        self,
    ) -> Tuple[
        torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
    ]:
        ds_list = []
        for ds_type in ["train", "val", "test"]:
            file_path = self.ckpt_dataset_path(ds_type)
            with open(file_path, "rb") as f:
                ds_list.append(pickle.load(f))  # nosec
        return ds_list
