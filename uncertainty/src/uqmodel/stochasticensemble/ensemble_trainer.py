import logging
import numpy as np
import torch
from torch.utils import tensorboard
from tqdm import tqdm
from .stochastic_mlc import StochasticMultiLayerClassifier
from .mlc import MultiLayerClassifier
from .loss import StochasticCrossEntropyLoss
from .checkpoint import EnsembleCheckpoint

logger = logging.getLogger(__name__)


class EnsembleTrainer(object):
    """
    A trainer for an ensemble classifiers of MLP.

    The trainer supports Checkpoint and learning scheduler.

    Reference:
        Learning scheduler
        1. https://discuss.pytorch.org/t/with-adam-optimizer-is-it-necessary-to-use-a-learning-scheduler/66477/3
        2. https://pytorch.org/docs/stable/optim.html
        3. Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019
        4. https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """

    def __init__(
        self,
        model_ensemble,
        criteria=None,
        lr_scheduler=None,
        max_iter=500,
        init_lr=1e-03,
        device=None,
        checkpoint=None,
        earlystopping=None,
        tensorboard_log=None,
        **kwargs,
    ):
        self.init_trainer(
            model_ensemble,
            criteria=criteria,
            lr_scheduler=lr_scheduler,
            max_iter=max_iter,
            init_lr=init_lr,
            device=device,
            checkpoint=checkpoint,
            earlystopping=earlystopping,
            tensorboard_log=tensorboard_log,
            **kwargs,
        )

    def init_trainer(
        self,
        model_ensemble,
        criteria=None,
        lr_scheduler=None,
        max_iter=500,
        init_lr=1e-03,
        device=None,
        checkpoint: EnsembleCheckpoint = None,
        earlystopping=None,
        tensorboard_log=None,
        **kwargs,
    ):
        self.tensorboard_logdir, self.tensorboard_logtag = None, None
        if isinstance(tensorboard_log, tuple) and len(tensorboard_log) == 2:
            self.tensorboard_logdir, self.tensorboard_logtag = tensorboard_log
        elif tensorboard_log:
            raise ValueError(
                "tensorboard_log must be None or a tuple of (logdir, logtag)"
            )

        self.max_iter = max_iter
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model_ensemble = model_ensemble.to(device)
        self.model_size = len(self.model_ensemble)
        self.checkpoint = checkpoint
        self.early_stopper = earlystopping
        self.begin_epoch = 0
        if "n_samples" in kwargs:
            self.n_samples = kwargs["n_samples"]
        for model in self.model_ensemble:
            if (
                isinstance(model, StochasticMultiLayerClassifier)
                and "ouput_log_sigma" not in kwargs
            ):
                raise ValueError(
                    "Stochastic ensemble model must provide ouput_log_sigma"
                )
        if "ouput_log_sigma" in kwargs:
            self.output_log_sigma = kwargs["ouput_log_sigma"]
            for model in self.model_ensemble:
                if self.output_log_sigma != model.output_log_sigma:
                    raise ValueError(
                        "Inconsistent output_log_sigma for Ensemble and individual stochastic models"
                    )

        self.criteria_ensemble = [None] * self.model_size
        for idx, model in enumerate(self.model_ensemble):
            if criteria is None:
                if isinstance(model, StochasticMultiLayerClassifier):
                    self.criteria_ensemble[idx] = StochasticCrossEntropyLoss(
                        self.n_samples, False
                    )  # not use log_sigma
                elif isinstance(model, MultiLayerClassifier):
                    self.criteria_ensemble[idx] = torch.nn.CrossEntropyLoss()
                else:
                    raise ValueError(
                        "unsupported classifier type {}".format(type(model))
                    )
            elif isinstance(criteria, list) and len(criteria) == self.model_size:
                self.criteria_ensemble[idx] = criteria[idx]
            else:
                raise ValueError("unsupported criteria, incorrect length or not a list")

        self.optimizer_ensemble = [
            torch.optim.Adam(model.parameters(), lr=init_lr)
            for model in self.model_ensemble
        ]
        if lr_scheduler is None:
            self.lr_scheduler_ensemble = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=100, T_mult=2
                )
                for optimizer in self.optimizer_ensemble
            ]
        elif isinstance(lr_scheduler, dict) and set(lr_scheduler.keys()) == set(
            ["T_0", "T_mult"]
        ):
            self.lr_scheduler_ensemble = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=lr_scheduler["T_0"], T_mult=lr_scheduler["T_mult"]
                )
                for optimizer in self.optimizer_ensemble
            ]
        else:
            raise ValueError(
                "lr_scheduler must be a dict with keys [T_0, T_mult] or None"
            )

    def _member_fit(
        self,
        model_idx,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        pin_memory=False,
    ):
        self.early_stopper.reset()

        n_train_batches, n_val_batches = len(train_dataloader), len(val_dataloader)

        tqdm_iterator = tqdm(
            range(self.begin_epoch, self.max_iter),
            desc="member {}".format(model_idx),
            position=model_idx + 1,
        )

        for epoch in tqdm_iterator:
            # train phase -- train an ensemble of models
            self.model_ensemble[model_idx].train()
            batch_train_loss = torch.zeros(n_train_batches).to(
                self.device, non_blocking=pin_memory
            )

            for batch_idx, train_batch in enumerate(train_dataloader):
                x, y = train_batch
                x = x.to(self.device, non_blocking=pin_memory)
                y = y.to(self.device, non_blocking=pin_memory)

                logits = self.model_ensemble[model_idx](x)
                loss = self.criteria_ensemble[model_idx](logits, y)

                self.optimizer_ensemble[model_idx].zero_grad()
                loss.backward()

                self.optimizer_ensemble[model_idx].step()
                self.lr_scheduler_ensemble[model_idx].step(
                    epoch + batch_idx / n_train_batches
                )

                batch_train_loss[batch_idx] = loss.item()
                if torch.isnan(loss):
                    raise ValueError(
                        "loss is nan at epoch {} for member model {}".format(
                            epoch, model_idx
                        )
                    )

            # validation phase -- validate an ensemble of models
            with torch.no_grad():
                batch_val_loss = torch.zeros(n_val_batches).to(
                    self.device, non_blocking=pin_memory
                )
                self.model_ensemble[model_idx].eval()  # set eval flag
                for batch_idx, val_batch in enumerate(val_dataloader):
                    x, y = val_batch
                    x = x.to(self.device, non_blocking=pin_memory)
                    y = y.to(self.device, non_blocking=pin_memory)
                    logits = self.model_ensemble[model_idx](x)
                    loss = self.criteria_ensemble[model_idx](logits, y)
                    batch_val_loss[batch_idx] = loss.item()
                    if torch.isnan(loss):
                        raise ValueError(
                            "loss is nan at epoch {} for member model {}".format(
                                epoch, model_idx
                            )
                        )

            total_train_loss = batch_train_loss.sum(dim=0)
            total_val_loss = batch_val_loss.sum(dim=0)
            mean_train_loss = total_train_loss / len(train_dataloader.dataset)
            mean_val_loss = total_val_loss / len(val_dataloader.dataset)
            logger.info(
                f"next learning rate for model -> {[lr.get_last_lr() for lr in self.lr_scheduler_ensemble]}"
            )
            logger.info(
                "epoch: {}, member model: {} mean_train_loss: {} mean_val_loss: {}".format(
                    epoch,
                    model_idx,
                    mean_train_loss,
                    mean_val_loss,
                )
            )
            if self.summary_writer:
                self.summary_writer.add_scalars(
                    "uq/{}/model/{}/loss".format(self.tensorboard_logtag, model_idx),
                    {"train_loss": mean_train_loss, "val_loss": mean_val_loss},
                    epoch,
                )
                lr = self.lr_scheduler_ensemble[model_idx]
                lr_dict = {"model_" + str(model_idx): np.array(lr.get_last_lr())}
                self.summary_writer.add_scalars(
                    "uq/{}/model/{}/lr".format(self.tensorboard_logtag, model_idx),
                    lr_dict,
                    epoch,
                )
            if self.checkpoint and self.checkpoint.min_loss_updated(
                model_idx, epoch, total_val_loss
            ):
                self.checkpoint.save_member_checkpoint(
                    epoch,
                    model_idx,
                    self.model_ensemble[model_idx],
                    self.optimizer_ensemble[model_idx],
                    self.lr_scheduler_ensemble[model_idx],
                    self.criteria_ensemble[model_idx],
                    self.checkpoint.min_member_losses[model_idx],
                )

            if self.early_stopper(total_val_loss):
                logger.info("EarlyStopping condition is met, stopping training ...")
                tqdm_iterator.close()
                break
        return epoch, total_train_loss, total_val_loss

    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        pin_memory=False,
    ):
        if self.tensorboard_logdir:
            self.summary_writer = tensorboard.SummaryWriter(self.tensorboard_logdir)
        else:
            self.summary_writer = None

        train_loss_ensemble = torch.zeros(self.model_size)
        val_loss_ensemble = torch.zeros(self.model_size)
        stop_epoches = torch.zeros(self.model_size)
        self.checkpoint.reset()
        for model_idx in tqdm(range(self.model_size), desc="ensemble", position=0):
            epoch, total_train_loss, total_val_loss = self._member_fit(
                model_idx, train_dataloader, val_dataloader, pin_memory
            )
            train_loss_ensemble[model_idx] = total_train_loss
            val_loss_ensemble[model_idx] = total_val_loss
            stop_epoches[model_idx] = epoch

        total_train_loss = train_loss_ensemble.sum(dim=0)
        total_val_loss = val_loss_ensemble.sum(dim=0)
        logger.info(
            "epoch: {}, train_loss: {}, val_loss: {}, total_train_loss: {} total_val_loss: {}".format(
                stop_epoches,
                train_loss_ensemble,
                val_loss_ensemble,
                total_train_loss,
                total_val_loss,
            )
        )

        if self.summary_writer:
            self.summary_writer.close()

        self.model_ensemble, total_loss = self.load_checkpoint()
        self.checkpoint.save_ensemble_meta(self.model_size, total_loss)
        return self.model_ensemble

    # def load_checkpoint(self):
    #     # optimizer_ensemble = [torch.optim.Adam(model.parameters(), lr=1e-03) for model in self.model_ensemble]
    #     # lr_scheduler_ensemble = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    #     #                               for optimizer in self.optimizer_ensemble]
    #     optimizer_ensemble = self.optimizer_ensemble
    #     lr_scheduler_ensemble = self.lr_scheduler_ensemble
    #     val_loss_ensemble = [0 for _ in self.model_ensemble]
    #     criteria_ensemble = self.criteria_ensemble

    #     (
    #         self.begin_epoch,
    #         self.model_ensemble,
    #         self.optimizer_ensemble,
    #         self.scheduler_ensemble,
    #         self.criteria_ensemble,
    #         _,
    #         min_total_loss,
    #         _,
    #     ) = self.checkpoint.load_checkpoint(
    #         self.model_ensemble,
    #         optimizer_ensemble,
    #         lr_scheduler_ensemble,
    #         criteria_ensemble,
    #         val_loss_ensemble,
    #     )
    #     self.checkpoint.min_total_loss = min_total_loss
    #     return self.model_ensemble

    def load_checkpoint(self):
        min_total_loss = 0
        optimizer_ensemble = self.optimizer_ensemble
        lr_scheduler_ensemble = self.lr_scheduler_ensemble
        criteria_ensemble = self.criteria_ensemble

        for model_idx, (model, optimizer, lr_scheduler, criteria) in enumerate(
            zip(
                self.model_ensemble,
                optimizer_ensemble,
                lr_scheduler_ensemble,
                criteria_ensemble,
            )
        ):
            logger.debug("loading checkpoint for member model {}".format(model_idx))
            assert isinstance(model_idx, int)
            assert isinstance(model, StochasticMultiLayerClassifier)

            (
                _,
                model,
                _,
                _,
                _,
                min_member_loss,
            ) = self.checkpoint.load_member_checkpoint(
                model_idx, model, optimizer, lr_scheduler, criteria
            )
            self.model_ensemble[model_idx] = model
            min_total_loss += min_member_loss
            logger.debug("loaded checkpoint for member model {}".format(model_idx))

        self.checkpoint.min_total_loss = min_total_loss
        logger.debug("loaded ensemble checkpoint")
        return self.model_ensemble, min_total_loss

    def retrain(self, ensemble):
        self.model_ensemble = ensemble
