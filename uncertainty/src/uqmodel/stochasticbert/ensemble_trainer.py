import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils import tensorboard
from tqdm import tqdm

from .data import BertExperimentDatasets
from .early_stopping import EarlyStopping
from .ensemble_bert import StochasticEnsembleBertClassifier
from .experiment import ExperimentConfig
from .logging_utils import get_global_logfilename, init_logging
from .loss import StochasticCrossEntropyLoss
from .stochastic_bert_mlc import StochasticBertBinaryClassifier

logger = logging.getLogger(__name__)


def get_dataset_size(
    dataset: Union[
        torch.utils.data.Dataset,
        torch.utils.data.Subset,
        torch.utils.data.ConcatDataset,
    ]
) -> int:
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return len(dataset)
    if isinstance(dataset, torch.utils.data.Subset):
        return len(dataset)
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return len(dataset)
    raise ValueError(
        f"expected torch.utils.data.TensorDataset, but encountered {type(dataset)}"
    )


def get_train_criteria(
    trainer_config: ExperimentConfig.TrainerConfig,
) -> torch.nn.CrossEntropyLoss:
    if trainer_config.criteria.loss_function == "cross_entropy_loss":
        criteria = torch.nn.CrossEntropyLoss()
    elif trainer_config.criteria.loss_function == "stochastic_cross_entropy_loss":
        criteria = StochasticCrossEntropyLoss(
            trainer_config.aleatoric_samples, use_log_sigma=False
        )
    else:
        raise ValueError(
            f"unsupported loss function {trainer_config.criteria.loss_function}"
        )
    return criteria


def get_train_optimizer(
    optimizer_config: ExperimentConfig.TrainerConfig.Optimizer,
    model: StochasticBertBinaryClassifier,
) -> Union[torch.optim.AdamW, torch.optim.Adam]:
    optimizer: Union[torch.optim.AdamW, torch.optim.Adam]
    if optimizer_config.optimizer == "Adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.init_lr)
    elif optimizer_config.optimizer == "AdamW":
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config.init_lr)
    else:
        raise ValueError(
            f"unsupported loss function {optimizer_config.optimizer}"
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
            f"unsupported lr_scheduler {lr_scheduler_config.scheduler}"
        )
    return lr_scheduler


class StochasticBertTrainer:
    summary_writer: Union[tensorboard.SummaryWriter, None]

    def __init__(
        self,
        model_idx: int,
        model: StochasticBertBinaryClassifier,
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
        self.criteria = get_train_criteria(trainer_config)
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
            f"uq/{self.tensorboard_logtag}/loss",
            {"train_loss": train_loss, "val_loss": val_loss},
            epoch,
        )
        self.summary_writer.add_scalars(
            f"uq/{self.tensorboard_logtag}/acc",
            {"train_acc": train_acc, "val_acc": val_acc},
            epoch,
        )
        self.summary_writer.add_scalars(
            f"uq/{self.tensorboard_logtag}/lr",
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
        logits_mu, _ = logits
        _, preds = torch.max(logits_mu, dim=1)
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
                    "loss is nan at epoch %d for training batch_idx %d",
                        epoch, batch_idx
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
                        "loss is nan at epoch %d for validation batch_idx %d",
                            epoch, batch_idx
                    )
                total_val_correct += self._get_num_correct(logits, targets)
                total_val_loss += loss.item()
            val_acc = total_val_correct / get_dataset_size(self.val_dataloader.dataset)
        return total_val_loss, val_acc

    def fit(self) -> StochasticBertBinaryClassifier:
        logger.info(
            "begin to train model for max_iter %d with train data size: %d",
            self.max_iter,
            len(self.train_dataloader.dataset),
        )
        old_usage = torch.cuda.memory_allocated(device=self.device)
        self.model = self.model.to(self.device, non_blocking=self.pin_memory)
        tqdm_iterator = tqdm(
            range(self.max_iter),
            desc=f"model_{self.model_idx}",
            position=1 + self.model_idx,
        )
        # tqdm_iterator = tqdm(range(self.max_iter))
        for epoch in tqdm_iterator:
            total_train_loss, train_acc = self._train_epoch(epoch)
            total_val_loss, val_acc = self._validation_epoch(epoch)
            mean_train_loss = total_train_loss / len(self.train_dataloader.dataset)
            mean_val_loss = total_val_loss / len(self.val_dataloader.dataset)

            logger.info(
                "epoch: %d, learning_rate: %s, total_train_loss: %f, total_val_loss: %f, train_acc: %f, val_acc: %f",
                    epoch,
                    str(self.lr_scheduler.get_last_lr()),
                    total_train_loss,
                    total_val_loss,
                    train_acc,
                    val_acc
            )

            self._log_tb_training_summary(
                epoch,
                mean_train_loss,
                mean_val_loss,
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
                tqdm_iterator.close()
                break

        if self.summary_writer:
            self.summary_writer.close()

        self.model.to(torch.device("cpu"), non_blocking=self.pin_memory)
        torch.cuda.empty_cache()
        new_usage = torch.cuda.memory_allocated(device=self.device)
        logger.info(
            "Model %d: CUDA memory allocation %d -> %d",
                self.model_idx, old_usage, new_usage
        )
        logger.info("completed training of member model %d", self.model_idx)
        return self.model


class StochasticEnsembleTrainer:
    """A trainer for an ensemble classifiers of Bert+MLP.

    The trainer supports Checkpoint and learning scheduler.
    """

    def __init__(
        self,
        model_ensemble: StochasticEnsembleBertClassifier,
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
        # # print('logfilename: {}'.format(logfilename))
        init_logging(logfilename, append=True)
        trainer = StochasticBertTrainer(
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
        for model_idx, model in tqdm(
            enumerate(self.ensemble_classifier),
            total=len(self.ensemble_classifier),
            position=0,
            desc="ensemble",
        ):
            # for model_idx,model in enumerate(self.ensemble_classifier):
            logger.info("begin to train model %d", model_idx)
            mp.set_sharing_strategy("file_system")
            # ctx = mp.get_context('spawn')
            # mp.set_sharing_strategy('file_descriptor')
            mp.set_start_method(method="forkserver", force=True)
            # manager = mp.Manager()
            # queue = manager.Queue()
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
                    f"{self.tensorboard_logtag}_m_{model_idx}",
                    self.device,
                    get_global_logfilename(),
                ),
            )
            p.start()
            p.join()
            logger.info("finished training model %d", model_idx)
        self.ensemble_classifier, total_loss = self.load_checkpoint()
        self.checkpoint.save_ensemble_meta(self.model_size, total_loss)
        if self.checkpoint.ckpt_tag:
            if self.datasets.pool_dataset:
                self.checkpoint.save_datasets(
                    self.datasets.train_dataset,
                    self.datasets.val_dataset,
                    self.datasets.test_dataset,
                    self.datasets.pool_dataset,
                )
            else:
                self.checkpoint.save_datasets(
                    self.datasets.train_dataset,
                    self.datasets.val_dataset,
                    self.datasets.test_dataset
                )
        return self.ensemble_classifier

    def load_checkpoint(self) -> Tuple[StochasticEnsembleBertClassifier, float]:
        min_total_loss: float = 0
        model_list: List[Union[StochasticBertBinaryClassifier, None]] = [None] * len(
            self.ensemble_classifier
        )
        for model_idx, model in enumerate(self.ensemble_classifier):
            logger.debug("loading checkpoint for member model %d", model_idx)
            assert isinstance(model_idx, int)
            assert isinstance(model, StochasticBertBinaryClassifier)
            criteria = get_train_criteria(self.trainer_config)
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
            logger.debug("loaded checkpoint for member model %d", model_idx)
        self.ensemble_classifier.model_ensemble = model_list

        self.checkpoint.min_total_loss = min_total_loss
        logger.debug("loaded ensemble checkpoint")
        return self.ensemble_classifier, min_total_loss


# import gc
# import logging
# import torch.multiprocessing as mp
# import numpy as np
# import torch
# from typing import List, Tuple, Union
# from torch.utils import tensorboard
# from tqdm import tqdm
# from .data import BertExperimentDatasets
# from .logging_utils import init_logging, get_global_logfilename
# from .early_stopping import EarlyStopping
# from .loss import StochasticCrossEntropyLoss
# from .ensemble_bert import StochasticEnsembleBertClassifier
# from .stochastic_bert_mlc import StochasticBertBinaryClassifier
# from .experiment import ExperimentConfig

# logger = logging.getLogger(__name__)


# def get_train_criteria(
#     trainer_config: ExperimentConfig.TrainerConfig,
# ) -> torch.nn.CrossEntropyLoss:
#     if trainer_config.criteria.loss_function == "cross_entropy_loss":
#         criteria = torch.nn.CrossEntropyLoss()
#     elif trainer_config.criteria.loss_function == "stochastic_cross_entropy_loss":
#         criteria = StochasticCrossEntropyLoss(
#             trainer_config.aleatoric_samples, use_log_sigma=False
#         )
#     else:
#         raise ValueError(
#             "unsupported loss function {}".format(trainer_config.criteria.loss_function)
#         )
#     return criteria


# def get_train_optimizer(
#     optimizer_config: ExperimentConfig.TrainerConfig.Optimizer,
#     model: StochasticBertBinaryClassifier,
# ) -> Union[torch.optim.AdamW, torch.optim.Adam]:
#     optimizer: Union[torch.optim.AdamW, torch.optim.Adam]
#     if optimizer_config.optimizer == "Adam":
#         optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_config.init_lr)
#     elif optimizer_config.optimizer == "AdamW":
#         optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_config.init_lr)
#     else:
#         raise ValueError(
#             "unsupported loss function {}".format(optimizer_config.optimizer)
#         )
#     return optimizer


# def get_train_lr_scheduler(
#     lr_scheduler_config: ExperimentConfig.TrainerConfig.LearningRateScheduler,
#     optimizer: Union[torch.optim.AdamW, torch.optim.Adam],
# ) -> torch.optim.lr_scheduler.StepLR:
#     if lr_scheduler_config.scheduler == "StepLR":
#         lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=lr_scheduler_config.step_size,
#             gamma=lr_scheduler_config.gamma,
#         )
#     else:
#         raise ValueError(
#             "unsupported lr_scheduler {}".format(lr_scheduler_config.scheduler)
#         )
#     return lr_scheduler


# class BertTrainer(object):
#     def __init__(
#         self,
#         model_idx: int,
#         model: torch.nn.Module,
#         datasets: BertExperimentDatasets,
#         lr_params: dict,
#         max_iter: int,
#         batch_size: int,
#         num_workers: int,
#         pin_memory: bool = False,
#         early_stopper: EarlyStopping = None,
#         tensorboard_logdir: str = None,
#         tensorboard_logtag: str = None,
#         n_aleatoric_samples: int = 100,
#         device: torch.DeviceObjType = None,
#     ):
#         logger.debug("creating BertTrainer")
#         self.model_idx = model_idx
#         self.model = model
#         self.n_aleatoric_samples = n_aleatoric_samples
#         self.criteria = StochasticCrossEntropyLoss(self.n_aleatoric_samples, False)
#         self.optimizer = torch.optim.AdamW(
#             self.model.parameters(), lr=lr_params["init_lr"]
#         )
#         self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             self.optimizer, step_size=lr_params["step_size"], gamma=lr_params["gamma"]
#         )
#         self.max_iter = max_iter
#         self.train_dataloader = torch.utils.data.DataLoader(
#             datasets.run_dataset,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )
#         self.val_dataloader = torch.utils.data.DataLoader(
#             datasets.val_dataset,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#         )
#         self.pin_memory = pin_memory
#         self.early_stopper = early_stopper
#         self.checkpoint = datasets.ckpt
#         self.tensorboard_logdir = tensorboard_logdir
#         self.tensorboard_logtag = tensorboard_logtag
#         self.summary_writer = None
#         self.device = device

#     def _lr_scheduler_inner_step(self, n_batches: int, batch_idx: int, epoch: int):
#         if isinstance(
#             self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#         ):
#             self.lr_scheduler.step(epoch + batch_idx / n_batches)

#     def _lr_scheduler_outer_step(self):
#         if not isinstance(
#             self.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#         ):
#             self.lr_scheduler.step()

#     def _log_tb_training_summary(
#         self, epoch: int, train_loss: float, val_loss: float, lr: float
#     ):
#         if not self.tensorboard_logdir:
#             return

#         if not self.summary_writer:
#             self.summary_writer = tensorboard.SummaryWriter(self.tensorboard_logdir)

#         self.summary_writer.add_scalars(
#             "uq/{}/loss".format(self.tensorboard_logtag),
#             {"train_loss": train_loss, "val_loss": val_loss},
#             epoch,
#         )
#         self.summary_writer.add_scalars(
#             "uq/{}/lr".format(self.tensorboard_logtag),
#             {"learning_rate": np.array(lr)},
#             epoch,
#         )

#     def _batch_train_step(self, batch: Tuple):
#         input_ids, attention_mask, targets = batch
#         input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
#         attention_mask = attention_mask.to(self.device, non_blocking=self.pin_memory)
#         targets = targets.to(self.device, non_blocking=self.pin_memory)

#         logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         loss = self.criteria(logits, targets)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return loss

#     def _train_epoch(self, epoch: int):
#         n_train_batches = len(self.train_dataloader)
#         train_losses = torch.zeros(n_train_batches)
#         self.model.train()
#         for batch_idx, batch in enumerate(self.train_dataloader):
#             loss = self._batch_train_step(batch)
#             train_losses[batch_idx] = loss.item()
#             if torch.isnan(train_losses[batch_idx]):
#                 logger.warning(
#                     "loss is nan at epoch {} for training batch_idx {}".format(
#                         epoch, batch_idx
#                     )
#                 )
#             self._lr_scheduler_inner_step(n_train_batches, batch_idx, epoch)
#         self._lr_scheduler_outer_step()
#         return train_losses

#     def _batch_validation_step(self, batch: Tuple):
#         input_ids, attention_mask, targets = batch
#         input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
#         attention_mask = attention_mask.to(self.device, non_blocking=self.pin_memory)
#         targets = targets.to(self.device, non_blocking=self.pin_memory)
#         logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         loss = self.criteria(logits, targets)
#         return loss

#     def _validation_epoch(self, epoch: int):
#         n_val_batches = len(self.val_dataloader)
#         val_losses = torch.zeros(n_val_batches)
#         with torch.no_grad():
#             self.model.eval()
#             for batch_idx, batch in enumerate(self.val_dataloader):
#                 loss = self._batch_validation_step(batch)
#                 val_losses[batch_idx] = loss.item()
#                 if torch.isnan(val_losses[batch_idx]):
#                     logger.warning(
#                         "loss is nan at epoch {} for validation batch_idx {}".format(
#                             epoch, batch_idx
#                         )
#                     )
#         return val_losses

#     def fit(self):
#         logger.info("begin to train model for max_iter {}".format(self.max_iter))
#         old_usage = torch.cuda.memory_allocated(device=self.device)
#         self.model = self.model.to(self.device, non_blocking=self.pin_memory)
#         for epoch in tqdm(range(self.max_iter), desc='member {}'.format(self.model_idx)):
#             train_losses = self._train_epoch(epoch)
#             val_losses = self._validation_epoch(epoch)

#             total_train_loss = torch.sum(train_losses)
#             total_val_loss = torch.sum(val_losses)
#             logger.info(
#                 "epoch: {}, learning_rate: {}, total_train_loss: {} total_val_loss: {}".format(
#                     epoch,
#                     self.lr_scheduler.get_last_lr(),
#                     total_train_loss,
#                     total_val_loss,
#                 )
#             )

#             self._log_tb_training_summary(
#                 epoch, total_train_loss, total_val_loss, self.lr_scheduler.get_last_lr()
#             )

#             if self.checkpoint and self.checkpoint.min_loss_updated(
#                 self.model_idx, epoch, total_val_loss
#             ):
#                 self.checkpoint.save_member_checkpoint(
#                     epoch,
#                     self.model_idx,
#                     self.model,
#                     self.optimizer,
#                     self.lr_scheduler,
#                     self.criteria,
#                     self.checkpoint.min_member_losses[self.model_idx]
#                 )

#             if self.early_stopper(total_val_loss):
#                 logger.info("EarlyStopping condition is met, stopping training ...")
#                 break

#         if self.summary_writer:
#             self.summary_writer.close()

#         self.model.to(torch.device("cpu"), non_blocking=self.pin_memory)
#         torch.cuda.empty_cache()
#         new_usage = torch.cuda.memory_allocated(device=self.device)
#         logger.info(
#             "Model {}: CUDA memory allocation {} -> {}".format(
#                 self.model_idx, old_usage, new_usage
#             )
#         )
#         return self.model


# class EnsembleTrainer(object):
#     """
#     A trainer for an ensemble classifiers of Bert+MLP.

#     The trainer supports Checkpoint and learning scheduler.
#     """

#     def __init__(
#         self,
#         model_ensemble: StochasticEnsembleBertClassifier,
#         datasets: BertExperimentDatasets,
#         lr_scheduler_params: dict,
#         init_lr: float = 1e-03,
#         max_iter: int = 500,
#         num_workers: int = 1,
#         batch_size: int = 32,
#         pin_memory: bool = False,
#         device=None,
#         earlystopping=None,
#         tensorboard_log=None,
#         n_aleatoric_samples=100,
#     ):
#         self._init_tensorboard(tensorboard_log)
#         self._init_compute_device(device)
#         self.lr_params = {**lr_scheduler_params, "init_lr": init_lr}
#         self.max_iter = max_iter
#         self.num_workers = num_workers
#         self.batch_size = batch_size
#         self.pin_memory = pin_memory
#         self.ensemble_classifier = model_ensemble
#         self.model_size = len(self.ensemble_classifier)
#         self.datasets = datasets
#         self.checkpoint = datasets.ckpt
#         self.early_stopper = earlystopping
#         self.n_aleatoric_samples = n_aleatoric_samples

#     def _init_tensorboard(self, tensorboard_log):
#         self.tensorboard_logdir, self.tensorboard_logtag = None, None
#         if isinstance(tensorboard_log, tuple) and len(tensorboard_log) == 2:
#             self.tensorboard_logdir, self.tensorboard_logtag = tensorboard_log
#         elif tensorboard_log:
#             raise ValueError(
#                 "tensorboard_log must be None or a tuple of (logdir, logtag)"
#             )

#     def _init_compute_device(self, device):
#         if device is None:
#             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device

#     def _release_inner_loop_train_memory(self, *args):
#         cpu_args = []
#         for arg in args:
#             if hasattr(arg, "to"):
#                 cpu = arg.to("cpu")
#                 cpu_args.append(cpu)
#             del arg
#         torch.cuda.empty_cache()
#         gc.collect()
#         return cpu_args

#     def _release_train_memory(self, *args):
#         for arg in args:
#             if hasattr(arg, "to"):
#                 arg.to("cpu")
#             del arg
#         torch.cuda.empty_cache()
#         gc.collect()

#     def _trainer_fit(
#         self,
#         model_idx,
#         model,
#         datasets,
#         lr_params,
#         max_iter,
#         batch_size,
#         num_workers,
#         pin_memory,
#         early_stopper,
#         tensorboard_logdir,
#         tensorboard_logtag,
#         n_aleatoric_samples,
#         device,
#         logfilename,
#     ):
#         init_logging(logfilename, append=True)
#         trainer = BertTrainer(
#             model_idx,
#             model,
#             datasets,
#             lr_params,
#             max_iter,
#             batch_size,
#             num_workers,
#             pin_memory,
#             early_stopper,
#             tensorboard_logdir,
#             tensorboard_logtag,
#             n_aleatoric_samples,
#             device,
#         )
#         trainer.fit()

#     def fit(self):
#         for model_idx, model in enumerate(self.ensemble_classifier):
#             logger.info("begin to train model {}".format(model_idx))
#             mp.set_sharing_strategy("file_system")
#             ctx = mp.get_context("spawn")
#             p = ctx.Process(
#                 target=self._trainer_fit,
#                 args=(
#                     model_idx,
#                     model,
#                     self.datasets,
#                     self.lr_params,
#                     self.max_iter,
#                     self.batch_size,
#                     self.num_workers,
#                     self.pin_memory,
#                     self.early_stopper,
#                     self.tensorboard_logdir,
#                     self.tensorboard_logtag,
#                     self.n_aleatoric_samples,
#                     self.device,
#                     get_global_logfilename(),
#                 ),
#             )
#             p.start()
#             p.join()
#             logger.info("finished training model {}".format(model_idx))
#         self.ensemble_classifier, min_total_loss = self.load_checkpoint()
#         self.checkpoint.save_ensemble_meta(
#             0, self.model_size, min_total_loss
#         )
#         return self.ensemble_classifier

#     def load_checkpoint(self) -> Tuple[StochasticEnsembleBertClassifier, float]:
#         min_total_loss: float = 0
#         model_list: List[Union[StochasticBertBinaryClassifier, None]] = [None] * len(
#             self.ensemble_classifier
#         )
#         for model_idx, model in enumerate(self.ensemble_classifier):
#             logger.debug("loading checkpoint for member model {}".format(model_idx))
#             assert isinstance(model_idx, int)
#             assert isinstance(model, StochasticBertBinaryClassifier)
#             criteria = get_train_criteria(self.trainer_config)
#             optimizer = get_train_optimizer(self.trainer_config.optimizer, model)
#             lr_scheduler = get_train_lr_scheduler(
#                 self.trainer_config.lr_scheduler, optimizer
#             )
#             lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
#             (
#                 _,
#                 model,
#                 optimizer,
#                 lr_scheduler,
#                 criteria,
#                 min_member_loss,
#             ) = self.checkpoint.load_member_checkpoint(
#                 model_idx, model, optimizer, lr_scheduler, criteria
#             )
#             model_list[model_idx] = model
#             min_total_loss += min_member_loss
#             logger.debug("loaded checkpoint for member model {}".format(model_idx))
#         self.ensemble_classifier.model_ensemble = model_list

#         self.checkpoint.min_total_loss = min_total_loss
#         logger.debug("loaded ensemble checkpoint")
#         return self.ensemble_classifier, min_total_loss
