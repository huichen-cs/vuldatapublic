import logging
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils import tensorboard
from tqdm import tqdm
from typing import List, Tuple, Union

from .bert_mlc import BertBinaryClassifier
from .data import BertExperimentDatasets
from .early_stopping import EarlyStopping
from .experiment import ExperimentConfig
from .ensemble_bert import EnsembleBertClassifier
from .logging_utils import init_logging, get_global_logfilename

logger = logging.getLogger(__name__)


def get_dataset_size(dataset: torch.utils.data.Dataset) -> int:
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return len(dataset)
    else:
        raise ValueError(
            "expected torch.utils.data.TensorDataset, but encountered {}".format(
                type(dataset)
            )
        )


def get_train_criteria(
    criteria_config: ExperimentConfig.TrainerConfig.Criteria,
) -> torch.nn.CrossEntropyLoss:
    if criteria_config.loss_function == "cross_entropy_loss":
        criteria = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(
            "unsupported loss function {}".format(criteria_config.loss_function)
        )
    return criteria


def get_train_optimizer(
    optimizer_config: ExperimentConfig.TrainerConfig.Optimizer,
    model: BertBinaryClassifier,
) -> Union[torch.optim.AdamW, torch.optim.Adam]:
    optimizer: Union[torch.optim.AdamW, torch.optim.Adam]
    if optimizer_config.optimizer == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.init_lr)
    elif optimizer_config.optimizer == "AdamW":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config.init_lr)
    else:
        raise ValueError(
            "unsupported loss function {}".format(optimizer_config.optimizer)
        )
    return optimizer


def get_train_lr_scheduler(
    lr_scheduler_config: ExperimentConfig.TrainerConfig.LearningRateScheduler,
    optimizer: Union[torch.optim.AdamW, torch.optim.Adam],
) -> torch.optim.lr_scheduler.StepLR:
    if lr_scheduler_config.scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler_config.step_size,
            gamma=lr_scheduler_config.gamma,
        )
    else:
        raise ValueError(
            "unsupported lr_scheduler {}".format(lr_scheduler_config.scheduler)
        )
    return lr_scheduler


class BertTrainer(object):
    summary_writer: Union[tensorboard.SummaryWriter, None]

    def __init__(
        self,
        model_idx: int,
        model: BertBinaryClassifier,
        datasets: BertExperimentDatasets,
        trainer_config: ExperimentConfig.TrainerConfig,
        early_stopper: Union[EarlyStopping, None] = None,
        tensorboard_logdir: Union[str, None] = None,
        tensorboard_logtag: Union[str, None] = None,
        device: Union[torch.DeviceObjType, None] = None,
    ):
        logger.debug("creating BertTrainer")
        self.model_idx = model_idx
        self.model = model
        self.criteria = get_train_criteria(trainer_config.criteria)
        self.optimizer = get_train_optimizer(trainer_config.optimizer, self.model)
        self.lr_scheduler = get_train_lr_scheduler(
            trainer_config.lr_scheduler, self.optimizer
        )

        self.max_iter = trainer_config.max_iter
        self.train_dataloader = torch.utils.data.DataLoader(
            datasets.train_dataset,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_dataloader_workers,
            pin_memory=trainer_config.pin_memory,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            datasets.val_dataset,
            batch_size=trainer_config.batch_size,
            num_workers=trainer_config.num_dataloader_workers,
            pin_memory=trainer_config.pin_memory,
        )
        self.pin_memory = trainer_config.pin_memory
        self.early_stopper = early_stopper
        self.checkpoint = datasets.ckpt
        self.tensorboard_logdir = tensorboard_logdir
        self.tensorboard_logtag = tensorboard_logtag
        self.summary_writer = None
        self.device = device

    def _lr_scheduler_inner_step(self, n_batches: int, batch_idx: int, epoch: int):
        if isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            self.lr_scheduler.step(epoch + batch_idx / n_batches)

    def _lr_scheduler_outer_step(self):
        if not isinstance(
            self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        ):
            self.lr_scheduler.step()

    def _log_tb_training_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ):
        if not self.tensorboard_logdir:
            return

        if not self.summary_writer:
            self.summary_writer = tensorboard.SummaryWriter(self.tensorboard_logdir)

        self.summary_writer.add_scalars(
            "uq/{}/loss".format(self.tensorboard_logtag),
            {"train_loss": train_loss, "val_loss": val_loss},
            epoch,
        )
        self.summary_writer.add_scalars(
            "uq/{}/acc".format(self.tensorboard_logtag),
            {"train_acc": train_acc, "val_acc": val_acc},
            epoch,
        )
        self.summary_writer.add_scalars(
            "uq/{}/lr".format(self.tensorboard_logtag),
            {"learning_rate": np.array(lr)},
            epoch,
        )

    def _batch_train_step(self, input_ids, attention_mask, targets):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criteria(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, logits

    def _get_num_correct(self, logits, targets):
        _, preds = torch.max(logits, dim=1)
        correct = (preds == targets).sum().item()
        return correct

    def _train_epoch(self, epoch: int):
        n_train_batches = len(self.train_dataloader)
        total_train_loss = 0
        total_train_correct = 0
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
            attention_mask = attention_mask.to(
                self.device, non_blocking=self.pin_memory
            )
            targets = targets.to(self.device, non_blocking=self.pin_memory)
            loss, logits = self._batch_train_step(input_ids, attention_mask, targets)
            if torch.isnan(loss):
                logger.warning(
                    "loss is nan at epoch {} for training batch_idx {}".format(
                        epoch, batch_idx
                    )
                )
            total_train_loss += loss.item()
            total_train_correct += self._get_num_correct(logits, targets)
            self._lr_scheduler_inner_step(n_train_batches, batch_idx, epoch)
        self._lr_scheduler_outer_step()
        train_acc = total_train_correct / get_dataset_size(
            self.train_dataloader.dataset
        )
        return total_train_loss, train_acc

    def _batch_validation_step(self, input_ids, attention_mask, targets):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criteria(logits, targets)
        return loss, logits

    def _validation_epoch(self, epoch: int):
        with torch.no_grad():
            total_val_correct = 0
            total_val_loss = 0
            self.model.eval()
            for batch_idx, batch in enumerate(self.val_dataloader):
                input_ids, attention_mask, targets = batch
                input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
                attention_mask = attention_mask.to(
                    self.device, non_blocking=self.pin_memory
                )
                targets = targets.to(self.device, non_blocking=self.pin_memory)
                loss, logits = self._batch_validation_step(
                    input_ids, attention_mask, targets
                )
                if torch.isnan(loss):
                    logger.warning(
                        "loss is nan at epoch {} for validation batch_idx {}".format(
                            epoch, batch_idx
                        )
                    )
                total_val_correct += self._get_num_correct(logits, targets)
                total_val_loss += loss.item()
            val_acc = total_val_correct / get_dataset_size(self.val_dataloader.dataset)
        return total_val_loss, val_acc

    def fit(self):
        logger.info("begin to train model for max_iter {}".format(self.max_iter))
        old_usage = torch.cuda.memory_allocated(device=self.device)
        self.model = self.model.to(self.device, non_blocking=self.pin_memory)
        for epoch in tqdm(
            range(self.max_iter),
            desc="model_{}".format(self.model_idx),
            position=1 + self.model_idx,
        ):
            total_train_loss, train_acc = self._train_epoch(epoch)
            total_val_loss, val_acc = self._validation_epoch(epoch)

            logger.info(
                "epoch: {}, learning_rate: {}, total_train_loss: {} total_val_loss: {} train_acc: {}, val_acc: {}".format(
                    epoch,
                    self.lr_scheduler.get_last_lr(),
                    total_train_loss,
                    total_val_loss,
                    train_acc,
                    val_acc,
                )
            )

            self._log_tb_training_summary(
                epoch,
                total_train_loss,
                total_val_loss,
                train_acc,
                val_acc,
                self.lr_scheduler.get_last_lr(),
            )

            if self.checkpoint and self.checkpoint.min_loss_updated(
                self.model_idx, epoch, total_val_loss
            ):
                self.checkpoint.save_member_checkpoint(
                    epoch,
                    self.model_idx,
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.criteria,
                    self.checkpoint.min_member_losses[self.model_idx],
                )

            if self.early_stopper(total_val_loss):
                logger.info("EarlyStopping condition is met, stopping training ...")
                break

        if self.summary_writer:
            self.summary_writer.close()

        self.model.to(torch.device("cpu"), non_blocking=self.pin_memory)
        torch.cuda.empty_cache()
        new_usage = torch.cuda.memory_allocated(device=self.device)
        logger.info(
            "Model {}: CUDA memory allocation {} -> {}".format(
                self.model_idx, old_usage, new_usage
            )
        )
        return self.model


class EnsembleTrainer(object):
    """A trainer for an ensemble classifiers of Bert+MLP.

    The trainer supports Checkpoint and learning scheduler.
    """

    def __init__(
        self,
        model_ensemble: EnsembleBertClassifier,
        datasets: BertExperimentDatasets,
        trainer_config: ExperimentConfig.TrainerConfig,
        device: Union[torch.DeviceObjType, None] = None,
        earlystopping: Union[EarlyStopping, None] = None,
        tensorboard_log: Union[Tuple[str], None] = None,
    ):
        self._init_tensorboard(tensorboard_log)
        self._init_compute_device(device)
        self.datasets = datasets
        self.ensemble_classifier = model_ensemble
        self.model_size = len(self.ensemble_classifier)
        self.trainer_config = trainer_config
        self.checkpoint = datasets.ckpt
        self.early_stopper = earlystopping

    def _init_tensorboard(self, tensorboard_log):
        self.tensorboard_logdir, self.tensorboard_logtag = None, None
        if isinstance(tensorboard_log, tuple) and len(tensorboard_log) == 2:
            self.tensorboard_logdir, self.tensorboard_logtag = tensorboard_log
        elif tensorboard_log:
            raise ValueError(
                "tensorboard_log must be None or a tuple of (logdir, logtag)"
            )

    def _init_compute_device(self, device):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _trainer_fit(
        self,
        model_idx,
        model,
        datasets,
        trainer_config,
        early_stopper,
        tensorboard_logdir,
        tensorboard_logtag,
        device,
        logfilename,
    ):
        # print('logfilename: {}'.format(logfilename))
        init_logging(logfilename, append=True)
        trainer = BertTrainer(
            model_idx,
            model,
            datasets,
            trainer_config,
            early_stopper,
            tensorboard_logdir,
            tensorboard_logtag,
            device,
        )
        trainer.fit()

    def fit(self):
        # for model_idx,model in tqdm(enumerate(self.ensemble_classifier),
        #                             total=len(self.ensemble_classifier),
        #                             position=0,
        #                             desc='ensemble'):
        for model_idx, model in enumerate(self.ensemble_classifier):
            logger.info("begin to train model {}".format(model_idx))
            mp.set_sharing_strategy("file_system")
            # ctx = mp.get_context('spawn')
            mp.set_start_method(method="forkserver", force=True)
            ctx = mp.get_context("forkserver")
            p = ctx.Process(
                target=self._trainer_fit,
                args=(
                    model_idx,
                    model,
                    self.datasets,
                    self.trainer_config,
                    self.early_stopper,
                    self.tensorboard_logdir,
                    self.tensorboard_logtag + "_m_{}".format(model_idx),
                    self.device,
                    get_global_logfilename(),
                ),
            )
            p.start()
            p.join()
            logger.info("finished training model {}".format(model_idx))
        self.ensemble_classifier, total_loss = self.load_checkpoint()
        self.checkpoint.save_ensemble_meta(self.model_size, total_loss)
        return self.ensemble_classifier

    def load_checkpoint(self) -> Tuple[EnsembleBertClassifier, float]:
        min_total_loss: float = 0
        model_list: List[Union[BertBinaryClassifier, None]] = [None] * len(
            self.ensemble_classifier
        )
        for model_idx, model in enumerate(self.ensemble_classifier):
            assert isinstance(model_idx, int)
            assert isinstance(model, BertBinaryClassifier)
            criteria = get_train_criteria(self.trainer_config.criteria)
            optimizer = get_train_optimizer(self.trainer_config.optimizer, model)
            lr_scheduler = get_train_lr_scheduler(
                self.trainer_config.lr_scheduler, optimizer
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
            (
                _,
                model,
                optimizer,
                lr_scheduler,
                criteria,
                min_member_loss,
            ) = self.checkpoint.load_member_checkpoint(
                model_idx, model, optimizer, lr_scheduler, criteria
            )
            model_list[model_idx] = model
            min_total_loss += min_member_loss
        self.ensemble_classifier.model_ensemble = model_list

        self.checkpoint.min_total_loss = min_total_loss
        return self.ensemble_classifier, min_total_loss
