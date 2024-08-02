"""Algorithm:

- split train data into init_train, data_pool
- train with init_train (early stop using val_data)
- for portion in [0.1, 0.2, 0.3, 0.4, 0.5]:
    estimate uncertainty (predictive, aleatoric, epistemic) for data_pool
    for selection_method in [a. lowest aleatoric, highest epistemic;
                             b. highest aleatoric, lowest epistemic]
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
from copy import deepcopy
from datetime import datetime
from typing import Union

import torch
from uqmodel.stochasticbert.checkpoint import EnsembleCheckpoint
from uqmodel.stochasticbert.data import (
    BertExperimentDataLoaders,
    BertExperimentDatasets,
)
from uqmodel.stochasticbert.early_stopping import EarlyStopping
from uqmodel.stochasticbert.ensemble_bert import StochasticEnsembleBertClassifier
from uqmodel.stochasticbert.ensemble_trainer import StochasticEnsembleTrainer
from uqmodel.stochasticbert.eval_utils import EnsembleDisentangledUq
from uqmodel.stochasticbert.experiment import (
    ExperimentConfig,
    init_argparse,
    setup_reproduce,
)
from uqmodel.stochasticbert.logging_utils import init_logging

logger = logging.getLogger(__name__)


def get_active_learning_method_list():
    return [
        "ehal",
        "elah",
        "ehah",
        "elal",
        "aleh",
        "ahel",
        "aheh",
        "alel",
        "ehal_ratio",
        "elah_ratio",
        "ehal_max",
        "elah_max",
        "random",
    ]


def get_extended_argparser() -> argparse.ArgumentParser:
    parser = init_argparse()
    parser.add_argument(
        "-i",
        "--init_train_factor",
        help="initial training data factor, default = 5",
        default=5,
    )
    parser.add_argument(
        "-s",
        "--data_split_seed",
        help="data split generator seed, default = 1432",
        default=1432,
    )
    parser.add_argument(
        "-a",
        "--action",
        help=(
            "active learning action in "
            f"{list(['all','init']) + get_active_learning_method_list()}"
        ),
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
    if args.init_train_factor:
        config.init_train_factor = int(args.init_train_factor)
    if args.data_split_seed:
        config.data_split_seed = int(args.data_split_seed)
    return config


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
    logger.info("Experiment config: %s", config)
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
            "train/{get_datetime_jobid()}",
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
        (
            "N(train_dataset): %d "
            "N(val_dataset): %d, "
            "N(test_dataset): %d, "
            "N(run_dataset): %d, "
            "N(pool_dataset): %d",
        ),
        len(datasets.train_dataset),
        len(datasets.val_dataset),
        len(datasets.test_dataset),
        len(datasets.run_dataset),
        len(datasets.pool_dataset),
    )
    if load_trained:
        old_use_model = config.trainer.use_model
        config.trainer.use_model = "use_checkpoint"
    # train_datasets = deepcopy(datasets)
    # train_datasets.train_dataset = deepcopy(train_datasets.run_dataset)
    # ensemble = get_trained_model(config, train_datasets, device=config.device)
    ensemble = get_trained_model(config, datasets, device=config.device)
    if load_trained:
        config.trainer.use_model = old_use_model
    return ensemble


def compute_data_pool_uq_metrics(
    config: ExperimentConfig,
    ensemble: StochasticEnsembleBertClassifier,
    dataloaders: BertExperimentDataLoaders,
):
    logger.info(
        "Evaluating UQ for data pool of length %d",
        sum(len(batch[0]) for batch in dataloaders.pool_dataloader),
    )
    uq = EnsembleDisentangledUq(
        ensemble,
        dataloaders.pool_dataloader,
        config.trainer.aleatoric_samples,
        device=config.device,
    )
    (
        _,
        _,
        entropy_aleatoric,
        _,
        _,
        entropy_epistermic,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = uq.compute_uq()
    return (entropy_epistermic, entropy_aleatoric)


def update_run_dataset(
    method: str,
    run_datasets: BertExperimentDatasets,
    selection_size_list: list,
    i: int,
    entropy_epistermic,
    entropy_aleatoric,
):
    if "_ratio" in method:
        run_datasets.update_by_score(
            selection_size_list[i],
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{i}_{method}",
        )
    elif "_max" in method:
        run_datasets.update_by_intuition(
            selection_size_list[i],
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{i}_{method}",
        )
    elif method == "random":
        run_datasets.update_by_random(
            selection_size_list[i],
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{i}_{method}",
        )
    else:
        run_datasets.update(
            selection_size_list[i],
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{i}_{method}",
        )


def run_experiment(config: ExperimentConfig) -> dict:
    n_data_parts = 10
    experiment_datasets = BertExperimentDatasets(
        config,
        None,
        seed=config.data_split_seed,
        dataset_name="SAPDATA",
        init_train_factor=config.init_train_factor,
    )
    selection_size_list = [
        (
            len(experiment_datasets.pool_dataset) // n_data_parts
            if size < n_data_parts - 1
            else len(experiment_datasets.pool_dataset)
            - size * (len(experiment_datasets.pool_dataset) // n_data_parts)
        )
        for size in range(n_data_parts)
    ]

    logger.info("action to take: %s", config.action)

    run_datasets = deepcopy(experiment_datasets)
    run_datasets.train_dataset = deepcopy(run_datasets.run_dataset)
    if config.action in ["all", "init"]:
        logger.info(
            (
                "begin %s with len(train_dataset): %d, "
                "len(run_dataset): %d, len(pool_dataset): %d",
            ),
            "init",
            len(run_datasets.train_dataset),
            len(run_datasets.run_dataset),
            len(run_datasets.pool_dataset),
        )
        ensemble = get_trained_ensemble_model(config, run_datasets)
    else:
        # print('2-----------------------------------------------')
        ensemble = get_trained_ensemble_model(config, run_datasets, load_trained=True)
    logger.info("got initial ensemble model")

    entropy_epistermic, entropy_aleatoric = None, None
    if config.action in get_active_learning_method_list():
        # print('3-----------------------------------------------')
        experiment_dataloaders = BertExperimentDataLoaders(config, experiment_datasets)
        (entropy_epistermic, entropy_aleatoric) = compute_data_pool_uq_metrics(
            config, ensemble, experiment_dataloaders
        )
        logger.info("estimated UQ metrics for initial model")

    for method in get_active_learning_method_list():
        if config.action in (method, "all"):
            run_datasets = deepcopy(experiment_datasets)
            logger.info(
                (
                    "begin %s with len(train_dataset): %d, "
                    "len(run_dataset): %d, len(pool_dataset): %d",
                ),
                method,
                len(run_datasets.train_dataset),
                len(run_datasets.run_dataset),
                len(run_datasets.pool_dataset),
            )
            for i in range(n_data_parts):
                logger.info("run %s method for step %d", method, i)
                update_run_dataset(
                    method,
                    run_datasets,
                    selection_size_list,
                    i,
                    entropy_epistermic,
                    entropy_aleatoric,
                )
                run_datasets.train_dataset = deepcopy(run_datasets.run_dataset)
                logger.info(
                    (
                        "updated run_dataset for method %s with "
                        "len(train_dataset): %d, "
                        "len(run_dataset): %d, "
                        "len(pool_dataset): %d",
                    ),
                    method,
                    len(run_datasets.train_dataset),
                    len(run_datasets.run_dataset),
                    len(run_datasets.pool_dataset),
                )

                run_dataloaders = BertExperimentDataLoaders(config, run_datasets)
                logger.info(
                    (
                        "updated run_dataloaders for method %s with "
                        "N(run_dataloaders.run_dataloader): %d, "
                        "N(run_dataloaders.val_dataloder): %d, "
                        "N(run_dataloaders.pool_dataloder): %d",
                    ),
                    method,
                    sum(len(batch[0]) for batch in run_dataloaders.run_dataloader),
                    sum(len(batch[0]) for batch in run_dataloaders.val_dataloader),
                    sum(len(batch[0]) for batch in run_dataloaders.pool_dataloader),
                )
                ensemble = get_trained_ensemble_model(config, run_datasets)
                if i < n_data_parts - 1:
                    (
                        entropy_epistermic,
                        entropy_aleatoric,
                    ) = compute_data_pool_uq_metrics(config, ensemble, run_dataloaders)
                logger.info(
                    (
                        "done %s at %d with len(train_dataset): %d, "
                        "len(run_dataset): %d, len(val_dataset): %d, "
                        "len(pool_dataset): %d",
                    ),
                    method,
                    i,
                    len(run_datasets.train_dataset),
                    len(run_datasets.run_dataset),
                    len(run_datasets.val_dataset),
                    len(run_datasets.pool_dataset),
                )
    logger.info("done")


def resume_experiment(config: ExperimentConfig) -> dict:
    n_data_parts = 10
    experiment_datasets = BertExperimentDatasets(
        config,
        None,
        seed=config.data_split_seed,
        dataset_name="SAPDATA",
        init_train_factor=config.init_train_factor,
    )
    selection_size_list = [
        (
            len(experiment_datasets.pool_dataset) // n_data_parts
            if size < n_data_parts - 1
            else len(experiment_datasets.pool_dataset)
            - size * (len(experiment_datasets.pool_dataset) // n_data_parts)
        )
        for size in range(n_data_parts)
    ]

    logger.info("action to take: %s", config.action)

    checkpoint_idx, experiment_datasets, ensemble = get_checkpoint_index(
        config, rtn_checkpoint=True
    )
    experiment_datasets.run_dataset = deepcopy(experiment_datasets.train_dataset)

    logger.info("resume training from index %d+1", checkpoint_idx)
    experiment_dataloaders = BertExperimentDataLoaders(config, experiment_datasets)
    (entropy_epistermic, entropy_aleatoric) = compute_data_pool_uq_metrics(
        config, ensemble, experiment_dataloaders
    )
    logger.info("estimated UQ metrics for model at step %d", checkpoint_idx)

    for method in get_active_learning_method_list():
        if config.action not in (method, "all"):
            continue
        run_datasets = deepcopy(experiment_datasets)
        logger.info(
            (
                "begin %s with len(train_dataset): %d, "
                "len(run_dataset): %d, len(pool_dataset): %d",
            ),
            method,
            len(run_datasets.train_dataset),
            len(run_datasets.run_dataset),
            len(run_datasets.pool_dataset),
        )
        for i in range(checkpoint_idx+1, n_data_parts):
            logger.info("run %s method for step %d", method, i)
            update_run_dataset(
                method,
                run_datasets,
                selection_size_list,
                i,
                entropy_epistermic,
                entropy_aleatoric,
            )
            run_datasets.train_dataset = deepcopy(run_datasets.run_dataset)
            logger.info(
                (
                    "updated run_dataset for method %s with "
                    "len(train_dataset): %d, "
                    "len(run_dataset): %d, "
                    "len(pool_dataset): %d",
                ),
                method,
                len(run_datasets.train_dataset),
                len(run_datasets.run_dataset),
                len(run_datasets.pool_dataset),
            )

            run_dataloaders = BertExperimentDataLoaders(config, run_datasets)
            logger.info(
                (
                    "updated run_dataloaders for method %s with "
                    "N(run_dataloaders.run_dataloader): %d, "
                    "N(run_dataloaders.val_dataloder): %d, "
                    "N(run_dataloaders.pool_dataloder): %d",
                ),
                method,
                sum(len(batch[0]) for batch in run_dataloaders.run_dataloader),
                sum(len(batch[0]) for batch in run_dataloaders.val_dataloader),
                sum(len(batch[0]) for batch in run_dataloaders.pool_dataloader),
            )
            ensemble = get_trained_ensemble_model(config, run_datasets)
            if i < n_data_parts - 1:
                (
                    entropy_epistermic,
                    entropy_aleatoric,
                ) = compute_data_pool_uq_metrics(config, ensemble, run_dataloaders)
                logger.info(
                    (
                        "done %s at %d with len(train_dataset): %d, "
                        "len(run_dataset): %d, len(val_dataset): %d, "
                        "len(pool_dataset): %d",
                    ),
                    method,
                    i,
                    len(run_datasets.train_dataset),
                    len(run_datasets.run_dataset),
                    len(run_datasets.val_dataset),
                    len(run_datasets.pool_dataset),
                )
    logger.info("done")


def get_checkpoint_tag(checkpoint_idx: int, learn_method: str):
    return f"{checkpoint_idx}_{learn_method}"


def set_checkpoint_tag(checkpoint: EnsembleCheckpoint, tag: str):
    checkpoint.ckpt_tag = tag


def get_checkpoint_index(
    config: ExperimentConfig, rtn_checkpoint=False
):
    old_use_model = config.trainer.use_model
    if config.trainer.use_model != "use_checkpoint":
        print(
            f"use_model must be 'use_checkpoint', but got '{config.trainer.use_model}'"
        )
        config.trainer.use_model = "use_checkpoint"
        print("set config.trainer.use_model to 'use_checkpoint'")
    if config.action in ("init", "all"):
        raise ValueError(
            (
                "action must be one of "
                "['ehal', 'elah', 'ehah', 'elal', "
                "'aleh', 'ahel', 'aheh', 'alel', "
                "'ehal_ratio', 'elah_ratio', 'ehal_max', 'elah_max',"
                " 'random'], but got 'init' or 'all'"
            )
        )
    experiment_datasets = BertExperimentDatasets(
        config,
        None,
        seed=config.data_split_seed,
        dataset_name="SAPDATA",
        init_train_factor=config.init_train_factor,
    )
    checkpoint = EnsembleCheckpoint(
        config.model.ensemble_size,
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
        tag=None,
    )
    logger.info("Trying to load checkpoints ...")
    idx = 0
    ensemble = None
    while True:
        try:
            set_checkpoint_tag(checkpoint, get_checkpoint_tag(idx, config.action))
            dataset_list = checkpoint.load_datasets()
            if len(dataset_list) != 4:
                raise FileNotFoundError(
                    "dataset_list returned by "
                    "checkpoint.load_datasets() must have 4 elements "
                    "pool dataset not found"
                )
            experiment_datasets.train_dataset = dataset_list[0]
            experiment_datasets.val_dataset = dataset_list[1]
            experiment_datasets.test_dataset = dataset_list[2]
            experiment_datasets.pool_dataset = dataset_list[3]
            ensemble = get_trained_ensemble_model(config, experiment_datasets)
            idx += 1
        except FileNotFoundError as err:
            logger.info(
                "checkpoint %d not found due to type(err)=%s, err=%s",
                idx,
                type(err),
                err,
            )
            config.trainer.use_model = old_use_model
            if not rtn_checkpoint:
                return idx - 1
            else:
                return idx - 1, experiment_datasets, ensemble

if __name__ == "__main__":
    init_logging(__file__, append=False)

    exp_config = setup_experiment()

    if exp_config.action in ["init", "all"]:
        run_experiment(exp_config)
        exit(0)

    ckpt_idx_saved = get_checkpoint_index(exp_config)
    if ckpt_idx_saved >= 0:
        print(f"ckpt_idx_saved = {ckpt_idx_saved}, resume training")
        resume_experiment(exp_config)
    else:
        run_experiment(exp_config)
