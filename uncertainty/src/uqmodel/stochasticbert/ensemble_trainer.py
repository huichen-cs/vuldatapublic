import gc
import logging
import torch.multiprocessing as mp
import numpy as np
import torch
from torch.utils import tensorboard
from typing import Tuple
from uqmodel.stochasticbert.data import BertExperimentDatasets
from uqmodel.stochasticbert.logging_utils import init_logging, get_global_logfilename
from uqmodel.stochasticbert.train_utils import EarlyStopping
from uqmodel.stochasticbert.loss import StochasticCrossEntropyLoss
from uqmodel.stochasticbert.ensemble_bert import StochasticEnsembleBertClassifier

logger = logging.getLogger(__name__)


class BertTrainer(object):
    def __init__(
        self,
        model_idx: int,
        model: torch.nn.Module,
        datasets: BertExperimentDatasets,
        lr_params: dict,
        max_iter: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        early_stopper: EarlyStopping = None,
        tensorboard_logdir: str = None,
        tensorboard_logtag: str = None,
        n_aleatoric_samples: int = 100,
        device: torch.DeviceObjType = None,
    ):
        logger.debug("creating BertTrainer")
        self.model_idx = model_idx
        self.model = model
        self.n_aleatoric_samples = n_aleatoric_samples
        self.criteria = StochasticCrossEntropyLoss(self.n_aleatoric_samples, False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr_params["init_lr"]
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_params["step_size"], gamma=lr_params["gamma"]
        )
        self.max_iter = max_iter
        self.train_dataloader = torch.utils.data.DataLoader(
            datasets.run_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            datasets.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.pin_memory = pin_memory
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
        self, epoch: int, train_loss: float, val_loss: float, lr: float
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
            "uq/{}/lr".format(self.tensorboard_logtag),
            {"learning_rate": np.array(lr)},
            epoch,
        )

    def _batch_train_step(self, batch: Tuple):
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
        attention_mask = attention_mask.to(self.device, non_blocking=self.pin_memory)
        targets = targets.to(self.device, non_blocking=self.pin_memory)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criteria(logits, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _train_epoch(self, epoch: int):
        n_train_batches = len(self.train_dataloader)
        train_losses = torch.zeros(n_train_batches)
        self.model.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            loss = self._batch_train_step(batch)
            train_losses[batch_idx] = loss.item()
            if torch.isnan(train_losses[batch_idx]):
                logger.warning(
                    "loss is nan at epoch {} for training batch_idx {}".format(
                        epoch, batch_idx
                    )
                )
            self._lr_scheduler_inner_step(n_train_batches, batch_idx, epoch)
        self._lr_scheduler_outer_step()
        return train_losses

    def _batch_validation_step(self, batch: Tuple):
        input_ids, attention_mask, targets = batch
        input_ids = input_ids.to(self.device, non_blocking=self.pin_memory)
        attention_mask = attention_mask.to(self.device, non_blocking=self.pin_memory)
        targets = targets.to(self.device, non_blocking=self.pin_memory)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criteria(logits, targets)
        return loss

    def _validation_epoch(self, epoch: int):
        n_val_batches = len(self.val_dataloader)
        val_losses = torch.zeros(n_val_batches)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, batch in enumerate(self.val_dataloader):
                loss = self._batch_validation_step(batch)
                val_losses[batch_idx] = loss.item()
                if torch.isnan(val_losses[batch_idx]):
                    logger.warning(
                        "loss is nan at epoch {} for validation batch_idx {}".format(
                            epoch, batch_idx
                        )
                    )
        return val_losses

    def fit(self):
        logger.info("begin to train model for max_iter {}".format(self.max_iter))
        old_usage = torch.cuda.memory_allocated(device=self.device)
        self.model = self.model.to(self.device, non_blocking=self.pin_memory)
        for epoch in range(self.max_iter):
            train_losses = self._train_epoch(epoch)
            val_losses = self._validation_epoch(epoch)

            total_train_loss = torch.sum(train_losses)
            total_val_loss = torch.sum(val_losses)
            logger.info(
                "epoch: {}, learning_rate: {}, total_train_loss: {} total_val_loss: {}".format(
                    epoch,
                    self.lr_scheduler.get_last_lr(),
                    total_train_loss,
                    total_val_loss,
                )
            )
            print(
                "epoch: {}, learning_rate: {}, total_train_loss: {} total_val_loss: {}".format(
                    epoch,
                    self.lr_scheduler.get_last_lr(),
                    total_train_loss,
                    total_val_loss,
                )
            )

            self._log_tb_training_summary(
                epoch, total_train_loss, total_val_loss, self.lr_scheduler.get_last_lr()
            )

            if self.checkpoint and self.checkpoint.min_loss_updated(
                epoch, total_val_loss
            ):
                self.checkpoint.save_member_checkpoint(
                    self.model_idx,
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.criteria,
                    self.checkpoint.min_member_loss,
                    total_val_loss,
                )

            if self.early_stopper(total_val_loss):
                logger.info("EarlyStopping condition is met, stopping training ...")
                print("EarlyStopping condition is met, stopping training ...")
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
        print(
            "Model {}: CUDA memory allocation {} -> {}".format(
                self.model_idx, old_usage, new_usage
            )
        )
        return self.model


class EnsembleTrainer(object):
    """
    A trainer for an ensemble classifiers of Bert+MLP.

    The trainer supports Checkpoint and learning scheduler.
    """

    def __init__(
        self,
        model_ensemble: StochasticEnsembleBertClassifier,
        datasets: BertExperimentDatasets,
        lr_scheduler_params: dict,
        init_lr: float = 1e-03,
        max_iter: int = 500,
        num_workers: int = 1,
        batch_size: int = 32,
        pin_memory: bool = False,
        device=None,
        earlystopping=None,
        tensorboard_log=None,
        n_aleatoric_samples=100,
    ):
        self._init_tensorboard(tensorboard_log)
        self._init_compute_device(device)
        self.lr_params = {**lr_scheduler_params, "init_lr": init_lr}
        self.max_iter = max_iter
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.ensemble_classifier = model_ensemble
        self.model_size = len(self.ensemble_classifier)
        self.datasets = datasets
        self.checkpoint = datasets.ckpt
        self.early_stopper = earlystopping
        self.n_aleatoric_samples = n_aleatoric_samples

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

    def _release_inner_loop_train_memory(self, *args):
        cpu_args = []
        for arg in args:
            if hasattr(arg, "to"):
                cpu = arg.to("cpu")
                cpu_args.append(cpu)
            del arg
        torch.cuda.empty_cache()
        gc.collect()
        return cpu_args

    def _release_train_memory(self, *args):
        for arg in args:
            if hasattr(arg, "to"):
                arg.to("cpu")
            del arg
        torch.cuda.empty_cache()
        gc.collect()

    def _trainer_fit(
        self,
        model_idx,
        model,
        datasets,
        lr_params,
        max_iter,
        batch_size,
        num_workers,
        pin_memory,
        early_stopper,
        tensorboard_logdir,
        tensorboard_logtag,
        n_aleatoric_samples,
        device,
        logfilename,
    ):
        print("logfilename: {}".format(logfilename))
        init_logging(logfilename, append=True)
        trainer = BertTrainer(
            model_idx,
            model,
            datasets,
            lr_params,
            max_iter,
            batch_size,
            num_workers,
            pin_memory,
            early_stopper,
            tensorboard_logdir,
            tensorboard_logtag,
            n_aleatoric_samples,
            device,
        )
        trainer.fit()

    def fit(self):
        for model_idx, model in enumerate(self.ensemble_classifier):
            logger.info("begin to train model {}".format(model_idx))
            mp.set_sharing_strategy("file_system")
            ctx = mp.get_context("spawn")
            p = ctx.Process(
                target=self._trainer_fit,
                args=(
                    model_idx,
                    model,
                    self.datasets,
                    self.lr_params,
                    self.max_iter,
                    self.batch_size,
                    self.num_workers,
                    self.pin_memory,
                    self.early_stopper,
                    self.tensorboard_logdir,
                    self.tensorboard_logtag,
                    self.n_aleatoric_samples,
                    self.device,
                    get_global_logfilename(),
                ),
            )
            p.start()
            p.join()
            logger.info("finished training model {}".format(model_idx))
        self.ensemble_classifier, total_loss, min_total_loss = self.load_checkpoint()
        self.checkpoint.save_ensemble_meta(
            0, self.model_size, total_loss, min_total_loss
        )
        return self.ensemble_classifier

    def load_checkpoint(self):
        total_loss, min_total_loss = 0, 0
        model_ensemble = [None] * len(self.ensemble_classifier)
        for model_idx, model in enumerate(self.ensemble_classifier):
            criteria = StochasticCrossEntropyLoss(self.n_aleatoric_samples, False)
            optimizer = torch.optim.AdamW(model.parameters())
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
            (
                model,
                optimizer,
                lr_scheduler,
                criteria,
                loss,
                min_member_loss,
            ) = self.checkpoint.load_member_checkpoint(
                model_idx, model, optimizer, lr_scheduler, criteria
            )
            model_ensemble[model_idx] = model
            min_total_loss += min_member_loss
            total_loss += loss
        self.ensemble_classifier.model_ensemble = model_ensemble

        self.checkpoint.min_total_loss = min_total_loss
        self.checkpoint.total_loss = total_loss
        return self.ensemble_classifier, total_loss, min_total_loss
