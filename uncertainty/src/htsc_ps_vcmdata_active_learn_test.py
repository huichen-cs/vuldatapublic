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
    
    
TODO: data transofrmation (i.e., preprocessing) at present are fit with whole train datset. next 
    step, we should experiment with the scheme to fit with only the run_dataset
"""
import argparse
import logging
import os
import torch
from copy import deepcopy
from uqmodel.stochasticensemble.logging_utils import init_logging
from uqmodel.stochasticensemble.experiment_config import ExperimentConfig, init_argparse
from uqmodel.stochasticensemble.train_utils import get_trained_model
from uqmodel.stochasticensemble.active_learn import (
    PsFeatureExperimentDatasets,
    PsFeatureExperimentDataLoaders,
)
from uqmodel.stochasticensemble.eval_utils import EnsembleDisentangledUq
from uqmodel.stochasticensemble.ensemble_mlc import StochasticEnsembleClassifier
from uqmodel.stochasticensemble.experiment_config import (
    get_experiment_config,
    setup_reproduce,
)


logger = logging.getLogger(__name__)


def get_extended_argparser() -> argparse.ArgumentParser:
    parser = init_argparse()
    parser.add_argument(
        "-a",
        "--action",
        help="active learning action in {}".format(
            [
                "all",
                "init",
                "ehal",
                "elah",
                "ehah",
                "elal",
                "aleh",
                "ahel",
                "aheh",
                "alel",
            ]
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
    return config


def setup_experiment() -> ExperimentConfig:
    parser = get_extended_argparser()
    config = get_experiment_config(parser)

    if not os.path.exists(config.data.data_dir):
        raise ValueError(f"data_dir {config.data.data_dir} inaccessible")

    if config.reproduce:
        setup_reproduce(config.reproduce)

    if config.trainer.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    if os.cpu_count() < config.trainer.max_dataloader_workers:
        num_workers = os.cpu_count()
    else:
        num_workers = config.trainer.max_dataloader_workers
    config.num_workers = num_workers

    config = get_extended_args(config, parser)
    return config


def get_trained_ensemble_model(
    config: ExperimentConfig,
    datasets: PsFeatureExperimentDatasets,
    dataloaders: PsFeatureExperimentDataLoaders,
    load_trained: bool = False,
):
    logger.info(
        "N(run_dataloader): {}, N(val_dataloder): {}".format(
            sum([len(batch[0]) for batch in dataloaders.run_dataloader]),
            sum([len(batch[0]) for batch in dataloaders.val_dataloader]),
        )
    )
    if load_trained:
        old_use_model = config.trainer.use_model
        config.trainer.use_model = "use_checkpoint"
    ensemble = get_trained_model(
        config,
        "disentangle",
        dataloaders.run_dataloader,
        dataloaders.val_dataloader,
        datasets.ps_columns,
        datasets.ckpt,
        criteria=None,
        device=config.device,
    )
    if load_trained:
        config.trainer.use_model = old_use_model
    return ensemble


def compute_data_pool_uq_metrics(
    config: ExperimentConfig,
    ensemble: StochasticEnsembleClassifier,
    dataloaders: PsFeatureExperimentDataLoaders,
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
    experiment_datasets = PsFeatureExperimentDatasets(config, None)
    selection_size = int(len(experiment_datasets.pool_dataset) / 2 / n_data_parts)
    experiment_dataloaders = PsFeatureExperimentDataLoaders(config, experiment_datasets)

    if config.action in ["all", "init"]:
        logger.info(
            "begin {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                "init",
                len(experiment_datasets.run_dataset),
                len(experiment_datasets.pool_dataset),
            )
        )
        # print('begin {} with len(run_dataset): {}, len(pool_dataset): {}'
        #         .format('init', len(experiment_datasets.run_dataset), len(experiment_datasets.pool_dataset)))
        logger.info(
            "Init: N(experiment_dataloaders.run_dataloader): {}, N(experiment_dataloaders.val_dataloder): {}".format(
                sum([len(batch[0]) for batch in experiment_dataloaders.run_dataloader]),
                sum([len(batch[0]) for batch in experiment_dataloaders.val_dataloader]),
            )
        )
        # print('Init: N(experiment_dataloaders.run_dataloader): {}, N(experiment_dataloaders.val_dataloder): {}'
        #       .format(sum([len(batch[0]) for batch in experiment_dataloaders.run_dataloader]),
        #               sum([len(batch[0]) for batch in experiment_dataloaders.val_dataloader])))
        ensemble = get_trained_ensemble_model(
            config, experiment_datasets, experiment_dataloaders
        )
    else:
        ensemble = get_trained_ensemble_model(
            config, experiment_datasets, experiment_dataloaders, load_trained=True
        )
    (entropy_epistermic, entropy_aleatoric) = compute_data_pool_uq_metrics(
        config, ensemble, experiment_dataloaders
    )

    for method in ["ehal", "elah", "ehah", "elal", "aleh", "ahel", "aheh", "alel"]:
        if method == config.action or config.action == "all":
            run_datasets = deepcopy(experiment_datasets)
            logger.info(
                "begin {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                    method,
                    len(run_datasets.run_dataset),
                    len(run_datasets.pool_dataset),
                )
            )
            # print('begin {} with len(run_dataset): {}, len(pool_dataset): {}'
            #       .format(method, len(run_datasets.run_dataset), len(run_datasets.pool_dataset)))
            for i in range(n_data_parts):
                logger.info("run {} method for step {}".format(method, i))
                run_datasets.update(
                    selection_size,
                    entropy_epistermic,
                    entropy_aleatoric,
                    method,
                    "{}_{}".format(i, method),
                )
                logger.info(
                    "updated run_dataset for method {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                        method,
                        len(run_datasets.run_dataset),
                        len(run_datasets.pool_dataset),
                    )
                )
                # print('updated run_dataset for method {} with len(run_dataset): {}, len(pool_dataset): {}'
                #       .format(method, len(run_datasets.run_dataset), len(run_datasets.pool_dataset)))

                run_dataloaders = PsFeatureExperimentDataLoaders(config, run_datasets)
                logger.info(
                    "updated run_dataloaders for method {} with N(run_dataloaders.run_dataloader): {}, N(run_dataloaders.val_dataloder): {}".format(
                        method,
                        sum(
                            [len(batch[0]) for batch in run_dataloaders.run_dataloader]
                        ),
                        sum(
                            [len(batch[0]) for batch in run_dataloaders.val_dataloader]
                        ),
                    )
                )
                # print('updated run_dataloaders for method {} with N(run_dataloaders.run_dataloader): {}, N(run_dataloaders.val_dataloder): {}'
                #       .format(method,
                #               sum([len(batch[0]) for batch in run_dataloaders.run_dataloader]),
                #               sum([len(batch[0]) for batch in run_dataloaders.val_dataloader])))

                ensemble = get_trained_ensemble_model(
                    config, run_datasets, run_dataloaders
                )
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
                # print('done {} at {} with len(run_dataset): {}, len(pool_dataset): {}'
                #       .format(method, i, len(run_datasets.run_dataset), len(run_datasets.pool_dataset)))
    logger.info("done")
    # print('done')


if __name__ == "__main__":
    init_logging(logger, __file__, append=True)

    config = setup_experiment()

    run_experiment(config)
