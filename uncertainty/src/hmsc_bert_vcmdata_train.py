import logging
import os
import torch
from datetime import datetime

from uqmodel.bert.data import (
    BertExperimentDatasets,
)

# from uqmodel.predictive.focalloss_clcarwin import FocalLoss
from uqmodel.bert.checkpoint import EnsembleCheckpoint
from uqmodel.bert.early_stopping import EarlyStopping
from uqmodel.bert.experiment import (
    ExperimentConfig,
    get_experiment_config,
    setup_reproduce,
)
from uqmodel.bert.ensemble_bert import EnsembleBertClassifier
from uqmodel.bert.ensemble_trainer import EnsembleTrainer
from uqmodel.bert.logging_utils import init_logging


logger = logging.getLogger(__name__)


def get_datetime_jobid():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return date_time


# def get_dataset_stats(ds:torch.utils.data.Dataset) -> dict:
#     n_rows = len(ds)
#     n_cols = ds[0][0].shape[0]
#     x_0 = torch.stack([x for x,_,z in ds if z == 0])
#     x_1 = torch.stack([x for x,_,z in ds if z == 1])
#     n_rows_0 = len(x_0)
#     n_rows_1 = len(x_1)
#     return {'n_rows': n_rows, 'n_cols': n_cols, 'n_rows_0': n_rows_0, 'n_rows_1': n_rows_1}


# def compute_focalloss_alpha(train_dataset:torch.utils.data.Dataset,
#                             n_classes:int,
#                             imbalance_ratio:float,
#                             device:torch.DeviceObjType) -> float:
#     if n_classes != 2:
#         raise ValueError('implemented only for n_classes=2, unimplemented for n_classes={}'.format(n_classes))
#     stats = get_dataset_stats(train_dataset)
#     assert stats['n_rows_1']*imbalance_ratio == stats['n_rows_0']
#     focal_alpha = torch.tensor([stats['n_rows']/(n_classes*stats['n_rows_0']), stats['n_rows']/(n_classes*stats['n_rows_1'])])
#     logger.info('focal_alpha = {}'.format(focal_alpha))
#     focal_alpha = focal_alpha.to(device)
#     return focal_alpha

# def get_focal_loss(config:ExperimentConfig,
#                    train_dataset:torch.utils.data.Dataset):
#     focal_alpha = compute_focalloss_alpha(train_dataset,
#                                           config.model.num_classes,
#                                           config.data.imbalance_ratio,
#                                           config.device)
#     focal_gamma = config.trainer.criteria.focal_gamma
#     loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
#     logger.info('focal loss alpha: {}, focal loss gamma: {}'.format(focal_alpha, focal_gamma))
#     return loss


def get_trained_model(config: ExperimentConfig, datasets: BertExperimentDatasets):
    stopper = EarlyStopping(
        patience=config.trainer.early_stopping.patience,
        min_delta=config.trainer.early_stopping.min_delta,
    )

    ensemble = EnsembleBertClassifier(
        config.model.ensemble_size,
        num_classes=config.model.num_classes,
        neurons=config.model.num_neurons,
        dropouts=config.model.dropout_ratios,
        activation=config.model.activation,
        cache_dir=config.cache_dir,
    )

    # lr_scheduler_params = {
    #     'scheduler': config.trainer.lr_scheduler.scheduler,
    #     'step_size': config.trainer.lr_scheduler.step_size,
    #     'gamma': config.trainer.lr_scheduler.gamma
    # }

    ensemble_trainer = EnsembleTrainer(
        ensemble,
        datasets,
        trainer_config=config.trainer,
        device=config.device,
        earlystopping=stopper,
        tensorboard_log=(
            config.trainer.tensorboard_logdir,
            "train/{}".format(get_datetime_jobid()),
        ),
    )

    if config.trainer.use_model == "use_checkpoint":
        try:
            ensemble, _ = ensemble_trainer.load_checkpoint()
            logger.info("use the ensemble model from checkpoint, no training")
        except FileNotFoundError as err:
            logger.info("training the ensemble model from scratch")
            raise err
    else:
        logger.info("training the ensemble model from scratch")
        ensemble_trainer.fit()
        ensemble, _ = ensemble_trainer.load_checkpoint()
    return ensemble


def get_trained_ensemble_model(
    config: ExperimentConfig,
    datasets: BertExperimentDatasets,
    load_trained: bool = False,
):
    logger.info(
        "N(train_dataset): {}, N(val_dataset): {}".format(
            len(datasets.train_dataset), len(datasets.val_dataset)
        )
    )
    if load_trained:
        old_use_model = config.trainer.use_model
        config.trainer.use_model = "use_checkpoint"
    ensemble = get_trained_model(config, datasets)
    if load_trained:
        config.trainer.use_model = old_use_model
    return ensemble


def setup_experiment() -> ExperimentConfig:
    config = get_experiment_config()

    if not os.path.exists(config.data.data_dir):
        raise ValueError(f"data_dir {config.data.data_dir} inaccessible")

    if config.reproduce:
        setup_reproduce(config)

    if config.trainer.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    if os.cpu_count() < config.trainer.max_dataloader_workers:
        num_workers = os.cpu_count()
    else:
        num_workers = config.trainer.max_dataloader_workers
    config.trainer.num_dataloader_workers = num_workers

    return config


if __name__ == "__main__":
    init_logging(__file__, append=True)

    config = setup_experiment()

    ckpt = EnsembleCheckpoint(
        config.model.ensemble_size,
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
    )
    datasets = BertExperimentDatasets(config, tag=None)
    # dataloaders = BertExperimentDataLoaders(config, datasets, train=True)
    # focal_loss_ensemble = [get_focal_loss(config, datasets.train_dataset)
    #                        for _ in range(config.model.ensemble_size)]
    ensemble = get_trained_ensemble_model(config, datasets, load_trained=False)
    logger.info("Training completed")
