"""Algorithm.

- split train data into init_train, data_pool
- train with init_train (early stop using val_data)
- for portion in [0.1, 0.2, 0.3, 0.4, 0.5]:
    estimate uncertainty (predictive, aleatoric, epistemic) for data_pool
    for selection_method in [a. lowest aleatoric, highest epistemic; b. highest aleatoric, lowest epistemic]
        selected := select portion from data_pool using selection_method
        train_data := init_train + selected
        retrain with train_data (early stop using val_data)
        save model

- for portion in [0.1, 0.2, 0.3, 0.4, 0.5]:
    load model and data
    evaluate (computing uncertainties and F1)

"""
import argparse
import logging
import os
import torch
from copy import deepcopy
from datetime import datetime
from typing import Union
from uqmodel.stochasticbert.logging_utils import init_logging
from uqmodel.stochasticbert.data import (
    BertExperimentDatasets,
    BertExperimentDataLoaders,
)
from uqmodel.stochasticbert.experiment import (
    ExperimentConfig,
    init_argparse,
    setup_reproduce,
)
from uqmodel.stochasticbert.ensemble_trainer import StochasticEnsembleTrainer
from uqmodel.stochasticbert.eval_utils import EnsembleDisentangledUq
from uqmodel.stochasticbert.ensemble_bert import StochasticEnsembleBertClassifier
from uqmodel.stochasticbert.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


def get_experiment_config(
    parser: Union[argparse.ArgumentParser, None] = None,
) -> ExperimentConfig:
    if not parser:
        parser = init_argparse()
    args = parser.parse_args()
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"config file {args.config} inaccessible")
        config = ExperimentConfig(args.config)
    else:
        config = ExperimentConfig()
    logger.info(f"Experiment config: {config}")
    return config


def get_extended_argparser() -> argparse.ArgumentParser:
    parser = init_argparse()
    parser.add_argument(
        "-i",
        "--init_train_factor",
        help="initial training data factor, default = 2",
        default=2,
    )
    parser.add_argument(
        "-a",
        "--action",
        help="active learning action in {}".format(["all", "init", "ehal"]),
    )
    return parser


def get_extended_args(
    config: ExperimentConfig, parser: argparse.ArgumentParser = None
) -> ExperimentConfig:
    if not parser:
        parser = init_argparse()
    args = parser.parse_args()
    if args.action:
        config.action = args.action
    return config


def setup_experiment() -> ExperimentConfig:
    parser = get_extended_argparser()
    config = get_experiment_config(parser)

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

    config = get_extended_args(config, parser)
    return config


def get_datetime_jobid():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return date_time


def get_trained_model(
    config: ExperimentConfig, datasets: BertExperimentDatasets, device: torch.device
):
    stopper = EarlyStopping(
        patience=config.trainer.early_stopping.patience,
        min_delta=config.trainer.early_stopping.min_delta,
    )

    ensemble = StochasticEnsembleBertClassifier(
        config.model.ensemble_size,
        config.model.num_classes,
        config.model.num_neurons,
        config.model.dropout_ratios,
        config.model.activation,
        config.cache_dir,
    )

    ensemble_trainer = StochasticEnsembleTrainer(
        ensemble,
        datasets,
        config.trainer,
        device=device,
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
    if isinstance(ensemble, tuple):
        ensemble = ensemble[0]
    return ensemble


def get_trained_ensemble_model(
    config: ExperimentConfig,
    datasets: BertExperimentDatasets,
    load_trained: bool = False,
):
    logger.info(
        "N(run_dataset): {}, N(val_dataset): {}".format(
            len(datasets.run_dataset), len(datasets.pool_dataset)
        )
    )
    if load_trained:
        old_use_model = config.trainer.use_model
        config.trainer.use_model = "use_checkpoint"
    ensemble = get_trained_model(config, datasets, device=config.device)
    if load_trained:
        config.trainer.use_model = old_use_model
    return ensemble


def compute_data_pool_uq_metrics(
    config: ExperimentConfig,
    ensemble: StochasticEnsembleBertClassifier,
    dataloaders: BertExperimentDataLoaders,
):
    uq = EnsembleDisentangledUq(
        ensemble,
        dataloaders.pool_dataloader,
        config.trainer.aleatoric_samples,
        device=config.device,
    )
    (
        proba_std_aleatoric,
        proba_mean_aleatoric,
        entropy_aleatoric,
        proba_std_epistermic,
        proba_mean_epistermic,
        entropy_epistermic,
        proba_std,
        entropy_all,
        muinfo_all,
        mu_mean,
        sigma_aleatoric,
        sigma_epistermic,
    ) = uq.compute_uq()
    return (entropy_epistermic, entropy_aleatoric)


def run_experiment(config: ExperimentConfig) -> dict:
    n_data_parts = 5
    experiment_datasets = BertExperimentDatasets(config, None, dataset_name="VCMDATA")
    selection_size_list = [
        len(experiment_datasets.pool_dataset) // n_data_parts
        if size < n_data_parts - 1
        else len(experiment_datasets.pool_dataset)
        - size * (len(experiment_datasets.pool_dataset) // n_data_parts)
        for size in range(n_data_parts)
    ]

    logger.info("action to take: {}".format(config.action))

    if config.action in ["all", "init"]:
        # print('1++++++++++++++++++++++++++++++++++++++++++++++++')
        logger.info(
            "begin {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                "init",
                len(experiment_datasets.run_dataset),
                len(experiment_datasets.pool_dataset),
            )
        )
        ensemble = get_trained_ensemble_model(config, experiment_datasets)
    else:
        # print('2-----------------------------------------------')
        ensemble = get_trained_ensemble_model(
            config, experiment_datasets, load_trained=True
        )
    logger.info("got initial ensemble model")

    if config.action in ["ehal", "all"]:
        # print('3-----------------------------------------------')
        experiment_dataloaders = BertExperimentDataLoaders(config, experiment_datasets)
        (entropy_epistermic, entropy_aleatoric) = compute_data_pool_uq_metrics(
            config, ensemble, experiment_dataloaders
        )
        logger.info("estimated UQ metrics for initial model")

    for method in ["ehal"]:
        if method == config.action or config.action == "all":
            run_datasets = deepcopy(experiment_datasets)
            logger.info(
                "begin {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                    method,
                    len(run_datasets.run_dataset),
                    len(run_datasets.pool_dataset),
                )
            )
            for i in range(n_data_parts):
                logger.info("run {} method for step {}".format(method, i))
                run_datasets.update_by_score(
                    selection_size_list[i],
                    entropy_epistermic,
                    entropy_aleatoric,
                    method,
                    "{}_{}".format(i, method),
                    train=True,
                )
                logger.info(
                    "updated run_dataset for method {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                        method,
                        len(run_datasets.run_dataset),
                        len(run_datasets.pool_dataset),
                    )
                )

                run_dataloaders = BertExperimentDataLoaders(config, run_datasets)
                ensemble = get_trained_ensemble_model(config, run_datasets)
                if i < n_data_parts - 1:
                    (
                        entropy_epistermic,
                        entropy_aleatoric,
                    ) = compute_data_pool_uq_metrics(config, ensemble, run_dataloaders)
                logger.info(
                    "done {} at {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                        method,
                        i,
                        len(run_datasets.run_dataset),
                        len(run_datasets.pool_dataset),
                    )
                )

    logger.info("done")


if __name__ == "__main__":
    init_logging(__file__, append=True)

    config = setup_experiment()

    run_experiment(config)
