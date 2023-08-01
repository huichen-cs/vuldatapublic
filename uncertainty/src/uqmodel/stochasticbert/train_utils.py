import logging
import numpy as np
import os
import pickle  # nosec
import torch
from datetime import datetime
from typing import Tuple, List, Sequence, Union
from uqmodel.ensemble.experiment_config import ExperimentConfig
from uqmodel.ensemble.ps_data import (
    FeatureDataSet,
    StandardScalerTransform,
    PsFeatureDataCollection,
    PsFeatureExperimentDataBuilder,
    get_dataset_stats,
    get_dataloader_shape,
)
from uqmodel.ensemble.datashift import (
    DataShift,
    ShiftedFeatureDataSet,
    PortionShiftedFeatureDataSet,
)
from uqmodel.ensemble.stochastic_mlc import StochasticMultiLayerClassifier
from uqmodel.ensemble.ensemble_mlc import (
    StochasticEnsembleClassifier,
    EnsembleClassifier,
)
from uqmodel.ensemble.ensemble_trainer import EnsembleTrainer


logger = logging.getLogger(__name__)


class EarlyStopping(object):
    """
    Early stoper for model.

    Source:
        https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch/71999355#71999355
    """

    def __init__(self, patience=5, min_delta=0, min_loss=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_loss = min_loss

    def __call__(self, loss):
        logger.debug(
            f"EarlyStopping: min loss: {self.min_loss} loss: {loss} paitence: {self.patience} counter: {self.counter}"
        )
        if self.min_loss and ((loss - self.min_loss) >= self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        if not self.min_loss or loss < self.min_loss:
            self.min_loss = loss
        return self.early_stop


class EnsembleCheckpoint(object):
    """
    Save checkpoint.

    Reference:
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    """

    def __init__(self, checkpoint_dir_path, warmup_epochs=5, min_loss=None):
        self._ckpt_path = checkpoint_dir_path
        if not os.path.exists(self._ckpt_path):
            os.makedirs(self._ckpt_path)
        self._min_total_loss = min_loss
        self._warmup_epochs = warmup_epochs
        self._checkpointed = False

    @property
    def warmup_epochs(self):
        return self._warmup_epochs

    @warmup_epochs.setter
    def warmup_epochs(self, epochs):
        self._warmup_epochs = epochs

    @property
    def min_total_loss(self):
        return self._min_total_loss

    @min_total_loss.setter
    def min_total_loss(self, loss):
        self._min_total_loss = loss

    @property
    def ckpt_path(self):
        return self._ckpt_path

    def __call__(
        self,
        epoch,
        model_ensemble,
        optimizer_ensemble,
        scheduler_ensemble,
        criteria_ensemble,
        loss_ensemble,
        total_loss,
    ):
        logger.debug(
            f"Checkpoint: checking for min loss: {self._min_total_loss} loss: {total_loss}"
        )
        if epoch < self._warmup_epochs:
            if self._min_total_loss and total_loss < self._min_total_loss:
                self._min_total_loss = total_loss
            elif not self._min_total_loss:
                self._min_total_loss = total_loss
            return
        if (self._min_total_loss and total_loss < self._min_total_loss) or (
            not self._checkpointed
        ):
            self.save_checkpoint(
                epoch,
                model_ensemble,
                optimizer_ensemble,
                scheduler_ensemble,
                criteria_ensemble,
                loss_ensemble,
                total_loss,
            )
            self._min_total_loss = total_loss
            self._checkpointed = True
        if not self._min_total_loss:
            self._min_total_loss = total_loss

    def save_checkpoint(
        self,
        epoch,
        model_ensemble,
        optimizer_ensemble,
        scheduler_ensemble,
        criteria_ensemble,
        loss_ensemble,
        total_loss,
    ):
        en_filepath = os.path.join(self._ckpt_path, "model_en")
        en_dict = {
            "epoch": epoch,
            "ensemble_size": len(model_ensemble),
            "total_loss": total_loss,
            "min_total_loss": self._min_total_loss,
        }
        torch.save(en_dict, en_filepath)
        for index, (model, optimizer, scheduler, criteria, loss) in enumerate(
            zip(
                model_ensemble,
                optimizer_ensemble,
                scheduler_ensemble,
                criteria_ensemble,
                loss_ensemble,
            )
        ):
            self.save_member_checkpoint(
                index, model, optimizer, scheduler, criteria, loss
            )
            logger.info(
                f"saved checkpoint for model {index} to {self._ckpt_path}, loss {self._min_total_loss} -> {total_loss}"
            )

    def save_member_checkpoint(
        self, index, model, optimizer, scheduler, criteria, loss
    ):
        file_path = os.path.join(self._ckpt_path, f"model_{index}")
        torch.save(
            {
                "index": index,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "criteria_state_dict": criteria.state_dict(),
                "loss": loss,
            },
            file_path,
        )

    def load_checkpoint(
        self,
        model_ensemble,
        optimizer_ensemble,
        scheduler_ensemble,
        criteria_ensemble,
        loss_ensemble,
    ):
        en_filepath = os.path.join(self._ckpt_path, "model_en")
        en_dict = torch.load(en_filepath)
        epoch = en_dict["epoch"]
        total_loss = en_dict["total_loss"]
        ensemble_size = en_dict["ensemble_size"]
        min_total_loss = en_dict["min_total_loss"]
        for i in range(ensemble_size):
            file_path = os.path.join(self._ckpt_path, f"model_{i}")
            member_checkpoint = torch.load(file_path)
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
        train_dataset: FeatureDataSet,
        val_dataset: FeatureDataSet,
        test_dataset: FeatureDataSet,
        ps_columns: List,
    ) -> None:
        for ds_type, ds in zip(
            ["train", "val", "test", "ps_columns"],
            [train_dataset, val_dataset, test_dataset, ps_columns],
        ):
            file_path = os.path.join(self._ckpt_path, f"dataset_{ds_type}.pickle")
            with open(file_path, "wb") as f:
                pickle.dump(ds, f)

    def load_datasets(
        self,
    ) -> Tuple[FeatureDataSet, FeatureDataSet, FeatureDataSet, List]:
        ds_list = []
        for ds_type in ["train", "val", "test", "ps_columns"]:
            file_path = os.path.join(self._ckpt_path, f"dataset_{ds_type}.pickle")
            with open(file_path, "rb") as f:
                ds_list.append(pickle.load(f))  # nosec
        return ds_list


def get_datetime_jobid():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return date_time


def build_datasets(config: ExperimentConfig, ckpt: EnsembleCheckpoint):
    if config.trainer.use_data == "try_checkpoint":
        try:
            train_dataset, val_dataset, test_dataset, ps_columns = ckpt.load_datasets()
            logger.info(
                "loaded train/val/test datasets from checkpoint at {}".format(
                    config.trainer.checkpoint.dir_path
                )
            )
        except FileNotFoundError:
            logger.info(
                "unable to load checkpoint, prepare data set with train/test ratios: {} and validation ratio: {}".format(
                    config.data.train_test_ratios, config.data.val_ratio
                )
            )
            dataset_builder = PsFeatureExperimentDataBuilder(
                PsFeatureDataCollection(config.data.data_dir),
                train_test_ratios=config.data.train_test_ratios,
                val_ratio=config.data.val_ratio,
                imbalance_ratio=config.data.imbalance_ratio,
                cve_sample_size=config.data.cve_sample_size,
                Transform=StandardScalerTransform,
                shuffle=True,
                seed=config.reproduce.data_sampling_seed,
            )
            train_dataset, val_dataset, test_dataset, ps_columns = (
                dataset_builder.train_train_dataset,
                dataset_builder.train_val_dataset,
                dataset_builder.test_dataset,
                dataset_builder.ps_columns,
            )
            logger.info(
                "dataset_builder.train_train_dataset stats: {}".format(
                    get_dataset_stats(train_dataset)
                )
            )
            logger.info(
                "dataset_builder.train_val_dataset stats: {}".format(
                    get_dataset_stats(val_dataset)
                )
            )
            logger.info(
                "dataset_builder.test_dataset stats: {}".format(
                    get_dataset_stats(test_dataset)
                )
            )
            ckpt.save_datasets(train_dataset, val_dataset, test_dataset, ps_columns)
    else:
        raise ValueError(
            "unsupported configuration option {} for config.trainer.use_model".format(
                config.trainer.use_model
            )
        )
    return train_dataset, val_dataset, test_dataset, ps_columns


def build_shifted_datasets(
    config: ExperimentConfig,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
):
    if not config.datashift:
        raise ValueError("configuration does not have data shift setup")

    datashift = DataShift.from_dict(config.datashift.param_dict)
    if config.data.shift_data_portion:
        (train_dataset, val_dataset, test_dataset) = (
            PortionShiftedFeatureDataSet(
                train_dataset, datashift, config.data.shift_data_portion
            ),
            PortionShiftedFeatureDataSet(
                val_dataset, datashift, config.data.shift_data_portion
            ),
            PortionShiftedFeatureDataSet(
                test_dataset, datashift, config.data.shift_data_portion
            ),
        )
        logger.info(
            "portion {} data are applied data shiftat sigma".format(
                config.datashift.sigma
            )
        )
    else:
        (train_dataset, val_dataset, test_dataset) = (
            ShiftedFeatureDataSet(train_dataset, datashift),
            ShiftedFeatureDataSet(val_dataset, datashift),
            ShiftedFeatureDataSet(test_dataset, datashift),
        )
        logger.info(
            "all data are applied data shift at sigma {}".format(config.datashift.sigma)
        )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(
    config: ExperimentConfig,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
):
    num_workers = (
        os.cpu_count()
        if os.cpu_count() < config.trainer.max_dataloader_workers
        else config.trainer.max_dataloader_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
        pin_memory=config.trainer.pin_memory,
    )
    if config.trainer.split_data == "sanity_check":
        # sanity check: set train == validation == test
        logger.info(
            "running sanity check, set train data = validation data = test data"
        )
        val_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=num_workers,
            pin_memory=config.trainer.pin_memory,
        )
        test_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=num_workers,
            pin_memory=config.trainer.pin_memory,
        )
    elif config.trainer.split_data == "train_val_test":
        logger.info(
            "running train_validation_test job, use validation and test data set"
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=num_workers,
            pin_memory=config.trainer.pin_memory,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.trainer.batch_size,
            num_workers=num_workers,
            pin_memory=config.trainer.pin_memory,
        )
    else:
        raise ValueError(
            "unsupported configuration option {} for config.trainer.split_data".format(
                config.trainer.use_data
            )
        )

    logger.info(
        "train/test dataloaders are ready: train.shape: {}, val.shape: {}, test.shape: {}".format(
            get_dataloader_shape(train_dataloader),
            get_dataloader_shape(val_dataloader),
            get_dataloader_shape(test_dataloader),
        )
    )
    return train_dataloader, val_dataloader, test_dataloader


def instantiate_ensemble_trainder(
    config: ExperimentConfig,
    model_type: str,
    ps_columns: Sequence,
    lr_scheduler: dict,
    stopper: EarlyStopping,
    ckpt: EnsembleCheckpoint,
    criteria: torch.nn.Module,
    device: torch.device,
):
    if model_type == "disentangle":
        output_log_sigma = False
        ensemble = StochasticEnsembleClassifier(
            StochasticMultiLayerClassifier,
            output_log_sigma,  # not use log_sigma
            config.model.ensemble_size,
            len(ps_columns),
            2,
            neurons=config.model.num_neurons,
            dropouts=config.model.dropout_ratios,
            activation=torch.nn.LeakyReLU(),
        )

        trainer = EnsembleTrainer(
            ensemble,
            criteria=criteria,
            lr_scheduler=lr_scheduler,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=stopper,
            tensorboard_log=(
                config.trainer.tensorboard_logdir,
                "train/{}".format(get_datetime_jobid()),
            ),
            n_samples=config.trainer.aleatoric_samples,
            ouput_log_sigma=output_log_sigma,
        )
    elif model_type == "predictive":
        ensemble = EnsembleClassifier(
            config.model.ensemble_size,
            len(ps_columns),
            2,
            neurons=config.model.num_neurons,
            dropouts=config.model.dropout_ratios,
            activation=torch.nn.LeakyReLU(),
        )

        trainer = EnsembleTrainer(
            ensemble,
            criteria=criteria,
            lr_scheduler=lr_scheduler,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=stopper,
            tensorboard_log=(
                config.trainer.tensorboard_logdir,
                "train/{}".format(get_datetime_jobid()),
            ),
        )
    else:
        raise ValueError(
            "model type {} is not in [disentangle, predictive]".format(model_type)
        )
    return ensemble, trainer


def re_initialize_ensemble_trainder_from_pretrain(
    ensemble: Union[StochasticEnsembleClassifier, EnsembleClassifier],
    trainer: EnsembleTrainer,
    config: ExperimentConfig,
    model_type: str,
    lr_scheduler: dict,
    stopper: EarlyStopping,
    ckpt: EnsembleCheckpoint,
    criteria: torch.nn.Module,
    device: torch.device,
):
    ensemble = trainer.load_checkpoint()
    if model_type == "disentangle":
        trainer.init_trainer(
            ensemble,
            criteria=criteria,
            lr_scheduler=lr_scheduler,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=stopper,
            tensorboard_log=(
                config.trainer.tensorboard_logdir,
                "train/{}".format(get_datetime_jobid()),
            ),
            n_samples=config.trainer.aleatoric_samples,
            output_log_sigma=False,
        )
    elif model_type == "predictive":
        trainer.init_trainer(
            ensemble,
            criteria=criteria,
            lr_scheduler=lr_scheduler,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=stopper,
            tensorboard_log=(
                config.trainer.tensorboard_logdir,
                "train/{}".format(get_datetime_jobid()),
            ),
        )
    else:
        raise ValueError(
            "model type {} is not in [disentangle, predictive]".format(model_type)
        )
    return ensemble, trainer


def get_trained_model(
    config: ExperimentConfig,
    model_type: str,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    ps_columns: Sequence,
    ckpt: EnsembleCheckpoint,
    criteria: torch.nn.Module,
    device: torch.device,
):
    lr_scheduler = {
        "T_0": config.trainer.lr_scheduler.T_0,
        "T_mult": config.trainer.lr_scheduler.T_mult,
    }
    stopper = EarlyStopping(
        patience=config.trainer.early_stopping.patience,
        min_delta=config.trainer.early_stopping.min_delta,
    )

    ensemble, trainer = instantiate_ensemble_trainder(
        config, model_type, ps_columns, lr_scheduler, stopper, ckpt, criteria, device
    )

    if config.trainer.use_model == "from_pretrain":
        try:
            trainer, ensemble = re_initialize_ensemble_trainder_from_pretrain(
                ensemble,
                trainer,
                config,
                model_type,
                lr_scheduler,
                stopper,
                ckpt,
                criteria,
                device,
            )
            logger.info(
                "retraining the ensemble model from the pretrained model (checkpoint)"
            )
            trainer.fit(
                train_dataloader, val_dataloader, pin_memory=config.trainer.pin_memory
            )
            ensemble = trainer.load_checkpoint()
        except FileNotFoundError as err:
            raise ValueError("no model exists for retrain") from err
    elif config.trainer.use_model == "from_scratch":
        logger.info("training the ensemble model from scratch")
        trainer.fit(
            train_dataloader, val_dataloader, pin_memory=config.trainer.pin_memory
        )
        ensemble = trainer.load_checkpoint()
    elif config.trainer.use_model == "try_checkpoint":
        try:
            ensemble = trainer.load_checkpoint()
            logger.info("use the ensemble model from checkpoint, no training")
        except FileNotFoundError:
            logger.info("training the ensemble model from scratch")
            trainer.fit(
                train_dataloader, val_dataloader, pin_memory=config.trainer.pin_memory
            )
            ensemble = trainer.load_checkpoint()
    elif config.trainer.use_model == "use_checkpoint":
        try:
            ensemble = trainer.load_checkpoint()
            logger.info("use the ensemble model from checkpoint, no training")
        except FileNotFoundError as err:
            logger.info("training the ensemble model from scratch")
            raise err
    else:
        raise ValueError("unknown train_method {}".format(config.trainer.train_method))
    return ensemble


class SamplingFeatureDataSet(FeatureDataSet):
    def __init__(self, dataset: FeatureDataSet, ratio: float):
        super(SamplingFeatureDataSet, self).__init__(
            dataset.df, dataset.feature_list, dataset.label_name, dataset.transform
        )
        self.ratio = ratio

        n_selected = np.ceil(len(self.df) * ratio / 2).astype(int)
        index_0_list = [
            i for i in range(len(self.df)) if self.df.iloc[i][self.label_name] == 0
        ]
        index_1_list = [
            i for i in range(len(self.df)) if self.df.iloc[i][self.label_name] == 1
        ]

        index_0_samples = np.random.choice(index_0_list, size=n_selected, replace=False)
        index_1_samples = np.random.choice(index_1_list, size=n_selected, replace=False)
        self.df = self.df.iloc[
            np.sort(np.concatenate((index_0_samples, index_1_samples), axis=0)), :
        ]

        self.transform.init(self.df, self.feature_list)

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
