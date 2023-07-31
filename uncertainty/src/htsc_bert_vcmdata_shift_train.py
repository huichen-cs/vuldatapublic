import logging
import os
from datetime import datetime

import torch

from uqmodel.shiftstochasticbert.data import BertExperimentDatasets
from uqmodel.shiftstochasticbert.datashift import IndependentGaussianNoiseDataShift
from uqmodel.shiftstochasticbert.early_stopping import EarlyStopping
from uqmodel.shiftstochasticbert.ensemble_bert import StochasticEnsembleBertClassifier
from uqmodel.shiftstochasticbert.ensemble_trainer import StochasticEnsembleTrainer
from uqmodel.shiftstochasticbert.experiment import (
    ExperimentConfig,
    get_experiment_config,
    setup_reproduce,
)
from uqmodel.shiftstochasticbert.logging_utils import init_logging

logger = logging.getLogger(__name__)


def get_datetime_jobid():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return date_time


def get_datashift_object(config: ExperimentConfig.DataShiftConfig):
    if config.type == "IndependentGaussianNoiseDataShift":
        shifter = IndependentGaussianNoiseDataShift(config.mu, config.sigma)
        return shifter
    else:
        raise ValueError("unsupported data quality shift type {}".format(config.type))


def get_trained_model(config: ExperimentConfig, datasets: BertExperimentDatasets):
    noiser = get_datashift_object(config.datashift)
    stopper = EarlyStopping(
        patience=config.trainer.early_stopping.patience,
        min_delta=config.trainer.early_stopping.min_delta,
    )
    ensemble = StochasticEnsembleBertClassifier(
        noiser,
        config.model.ensemble_size,
        num_classes=config.model.num_classes,
        neurons=config.model.num_neurons,
        dropouts=config.model.dropout_ratios,
        activation=config.model.activation,
        cache_dir=config.cache_dir,
    )

    ensemble_trainer = StochasticEnsembleTrainer(
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

    if config.trainer.criteria.loss_function != "stochastic_cross_entropy_loss":
        raise ValueError(
            "expected stochastic_cross_entropy_loss, but found {}".format(
                config.trainer.criteria.loss_function
            )
        )

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

    datasets = BertExperimentDatasets(config, tag=None, dataset_name="VCMDATA")
    ensemble = get_trained_ensemble_model(config, datasets, load_trained=False)
    logger.info("Training completed")
