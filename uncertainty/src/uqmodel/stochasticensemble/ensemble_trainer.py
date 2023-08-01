import logging
import numpy as np
import torch
from torch.utils import tensorboard
from uqmodel.stochasticensemble.stochastic_mlc import StochasticMultiLayerClassifier
from uqmodel.stochasticensemble.mlc import MultiLayerClassifier
from uqmodel.stochasticensemble.loss import StochasticCrossEntropyLoss

logger = logging.getLogger("EnsembleTrainer")


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
        checkpoint=None,
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

    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        pin_memory=False,
    ):
        n_train_batches, n_val_batches = len(train_dataloader), len(val_dataloader)

        if self.tensorboard_logdir:
            self.summary_writer = tensorboard.SummaryWriter(self.tensorboard_logdir)
        else:
            self.summary_writer = None

        for epoch in range(self.begin_epoch, self.max_iter):
            # train phase -- train an ensemble of models
            batch_train_loss = torch.zeros(n_train_batches, self.model_size).to(
                self.device, non_blocking=pin_memory
            )

            for batch_idx, train_batch in enumerate(train_dataloader):
                x, y = train_batch
                x = x.to(self.device, non_blocking=pin_memory)
                y = y.to(self.device, non_blocking=pin_memory)
                for model_idx in range(self.model_size):
                    self.model_ensemble[model_idx].train()  # set train flag

                    logits = self.model_ensemble[model_idx](x)

                    loss = self.criteria_ensemble[model_idx](logits, y)

                    self.optimizer_ensemble[model_idx].zero_grad()
                    loss.backward()

                    self.optimizer_ensemble[model_idx].step()
                    self.lr_scheduler_ensemble[model_idx].step(
                        epoch + batch_idx / n_train_batches
                    )

                    batch_train_loss[batch_idx, model_idx] = loss.item()
                    if torch.isnan(loss):
                        print("nan")

            # validation phase -- validate an ensemble of models
            with torch.no_grad():
                batch_val_loss = torch.zeros(n_val_batches, self.model_size).to(
                    self.device, non_blocking=pin_memory
                )
                for model_idx in range(self.model_size):
                    self.model_ensemble[model_idx].eval()  # set eval flag
                    for batch_idx, val_batch in enumerate(val_dataloader):
                        x, y = val_batch
                        x = x.to(self.device, non_blocking=pin_memory)
                        y = y.to(self.device, non_blocking=pin_memory)
                        logits = self.model_ensemble[model_idx](x)
                        loss = self.criteria_ensemble[model_idx](logits, y)
                        batch_val_loss[batch_idx, model_idx] = loss.item()
                        if torch.isnan(loss):
                            print("nan")

            train_loss_ensemble = batch_train_loss.mean(dim=0)
            val_loss_ensemble = batch_val_loss.mean(dim=0)
            total_train_loss = torch.sum(train_loss_ensemble)
            total_val_loss = torch.sum(val_loss_ensemble)
            logger.info(
                f"next learning rate for model -> {[lr.get_last_lr() for lr in self.lr_scheduler_ensemble]}"
            )
            logger.info(
                "epoch: {}, train_loss: {}, val_loss: {}, total_train_loss: {} total_val_loss: {}".format(
                    epoch,
                    train_loss_ensemble,
                    val_loss_ensemble,
                    total_train_loss,
                    total_val_loss,
                )
            )
            if self.summary_writer:
                self.summary_writer.add_scalars(
                    "uq/{}/loss".format(self.tensorboard_logtag),
                    {"train_loss": total_train_loss, "val_loss": total_val_loss},
                    epoch,
                )
                lr_dict = dict()
                for idx, lr in enumerate(self.lr_scheduler_ensemble):
                    lr_dict["model_" + str(idx)] = np.array(lr.get_last_lr())
                self.summary_writer.add_scalars(
                    "uq/{}/lr".format(self.tensorboard_logtag), lr_dict, epoch
                )
            if self.checkpoint:
                self.checkpoint(
                    epoch,
                    self.model_ensemble,
                    self.optimizer_ensemble,
                    self.lr_scheduler_ensemble,
                    self.criteria_ensemble,
                    val_loss_ensemble,
                    total_val_loss,
                )
            if self.early_stopper(total_val_loss):
                logger.info("EarlyStopping condition is met, stopping training ...")
                break
        if self.summary_writer:
            self.summary_writer.close()
        return self.model_ensemble

    def load_checkpoint(self):
        # optimizer_ensemble = [torch.optim.Adam(model.parameters(), lr=1e-03) for model in self.model_ensemble]
        # lr_scheduler_ensemble = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
        #                               for optimizer in self.optimizer_ensemble]
        optimizer_ensemble = self.optimizer_ensemble
        lr_scheduler_ensemble = self.lr_scheduler_ensemble
        val_loss_ensemble = [0 for _ in self.model_ensemble]
        criteria_ensemble = self.criteria_ensemble

        (
            self.begin_epoch,
            self.model_ensemble,
            self.optimizer_ensemble,
            self.scheduler_ensemble,
            self.criteria_ensemble,
            _,
            min_total_loss,
            _,
        ) = self.checkpoint.load_checkpoint(
            self.model_ensemble,
            optimizer_ensemble,
            lr_scheduler_ensemble,
            criteria_ensemble,
            val_loss_ensemble,
        )
        self.checkpoint.min_total_loss = min_total_loss
        return self.model_ensemble

    def retrain(self, ensemble):
        self.model_ensemble = ensemble
