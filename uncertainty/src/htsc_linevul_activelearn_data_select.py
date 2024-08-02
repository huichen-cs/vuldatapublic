"""Select LineVul training instances."""
import argparse
import logging
import os
from copy import deepcopy

import pandas as pd
import torch
import transformers
from uqmodel.stochasticbert.data import (
    BertExperimentDataLoaders,
    BertExperimentDatasets,
    TextClassificationDataset,
)
from uqmodel.stochasticbert.early_stopping import EarlyStopping
from uqmodel.stochasticbert.ensemble_bert import StochasticEnsembleBertClassifier
from uqmodel.stochasticbert.ensemble_trainer import StochasticEnsembleTrainer
from uqmodel.stochasticbert.eval_utils import EnsembleDisentangledUq
from uqmodel.stochasticbert.experiment import (
    ExperimentConfig,
    setup_reproduce,
)
from uqmodel.stochasticbert.logging_utils import init_logging

logger = logging.getLogger(__name__)


class LineVulExperimentConfig:
    """Additional configuration for LineVul data selection experiment."""
    def __init__(self):
        self.input_file = os.path.join("data", "linevul", "linevul_data.csv")
        self.output_file = os.path.join("data", "linevul", "linevul_selected_data.pt")
        self.data_select_method = "ehal_max"
        self.data_select_ratio = 0.4
        self.model_index = 0

    def from_cmdline(self, args:argparse.Namespace):
        self.input_file = args.linevul_input_file
        self.data_select_method = args.action
        self.data_select_ratio = args.linevul_ratio
        self.output_file = args.linevul_output_file
        self.model_index = args.model_index
        self.method = args.action
        return self


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


def load_trained_model(
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

    if config.trainer.use_model != "use_checkpoint":
        raise ValueError(
            (
                "Invalid value for config.trainer.use_model, "
                f"expected 'use_checkpoint', but saw {config.trainer.use_model}."
            )
        )

    try:
        ensemble, _ = ensemble_trainer.load_checkpoint()
        logger.info("loaded the ensemble model from checkpoint")
    except FileNotFoundError as err:
        logger.error("failed to load model type(err)=%s. err=%s", type(err), err)
        raise err

    if isinstance(ensemble, tuple):
        ensemble = ensemble[0]
    return ensemble


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-c config_file] [...]", description="Run an UQ experiment."
    )

    parser.add_argument("-c", "--config")
    parser.add_argument(
        "-i",
        "--init_train_factor",
        help="initial training data factor, default = 2",
        default=2,
    )
    parser.add_argument(
        "-s",
        "--data_split_seed",
        help="data split generator seed, default = 1432",
        default=2,
    )
    parser.add_argument(
        "-a",
        "--action",
        help=f"active learning action in {get_active_learning_method_list()}",
        default="ehal_max",
        type=str,
    )
    parser.add_argument(
        "--linevul_input_file", type=str, help="Input file containing data points"
    )
    parser.add_argument(
        "--linevul_output_file",
        type=str,
        help="Output file containing selected data points",
    )
    parser.add_argument(
        "--linevul_ratio",
        type=float,
        help="linevul data ratio",
    )
    parser.add_argument(
        "--model_index",
        type=int,
        help="linevul selection model index",
    )
    return parser


def setup_experiment() -> ExperimentConfig:
    parser = init_argparse()
    args = parser.parse_args()

    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"config file {args.config} inaccessible")
        config = ExperimentConfig(args.config)
    else:
        config = ExperimentConfig()
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
    logger.info("Experiment config: %s", config)
    
    if args.init_train_factor:
        config.init_train_factor = args.init_train_factor
    if args.data_split_seed:
        config.data_split_seed = args.data_split_seed

    linevul_config = LineVulExperimentConfig()
    linevul_config.from_cmdline(args)

    return config, linevul_config


def update_run_dataset(
    method: str,
    run_datasets: BertExperimentDatasets,
    selection_size: int,
    model_index: int,
    entropy_epistermic,
    entropy_aleatoric,
):
    if "_ratio" in method:
        run_datasets.update_by_score(
            selection_size,
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{model_index}_{method}",
        )
    elif "_max" in method:
        run_datasets.update_by_intuition(
            selection_size,
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{model_index}_{method}",
        )
    elif method == "random":
        run_datasets.update_by_random(
            selection_size,
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{model_index}_{method}",
        )
    else:
        run_datasets.update(
            selection_size,
            entropy_epistermic,
            entropy_aleatoric,
            method,
            f"{model_index}_{method}",
        )


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


def load_linevul_data(config, linevul_config):
    linevul_df = pd.read_csv(linevul_config.input_file, na_filter=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/codebert-base", cache_dir=config.cache_dir
    )
    linevul_datset = TextClassificationDataset(
        linevul_df,
        tokenizer,
        config.model.max_encoding_len,
        config.model.num_classes,
    )
    return linevul_datset

def get_checkpoint_tag(model_idx: int, learn_method: str):
    return f"{model_idx}_{learn_method}"

def run_data_select_experiment(config, linevul_config):
    if config.trainer.use_model != "use_checkpoint":
        logger.warning(
            (
                "Invalid value for config.trainer.use_model, "
                "expected 'use_checkpoint', but saw '%s'."
            ),
            config.trainer.use_model,
        )
        config.trainer.use_model = "use_checkpoint"
        logger.info("set config.trainer.use_model to '%s'", config.trainer.use_model)

    # load VCM data from checkpoint - not really being used
    experiment_datasets = BertExperimentDatasets(
        config,
        None,
        seed=config.data_split_seed,
        dataset_name="VCMDATA",
        init_train_factor=config.init_train_factor,
    )

    # load LineVul train data
    linevul_dataset = load_linevul_data(config, linevul_config)
    logger.info("loaded linevul dataset.")

    checkpint_tag = get_checkpoint_tag(
            linevul_config.model_index,
            linevul_config.data_select_method)
    experiment_datasets.update_checkpoint(checkpint_tag)
    logger.info("updated checkpoint's tag to %s", checkpint_tag) 
    
    ensemble = load_trained_model(config, experiment_datasets, config.device)
    logger.info("loaded ensemble for tag %s", checkpint_tag)

    run_datasets = deepcopy(experiment_datasets)
    run_datasets.pool_dataset = deepcopy(linevul_dataset)

    run_dataloaders = BertExperimentDataLoaders(config, run_datasets)
    logger.info(
        (
            "updated run_dataloaders for method %s with "
            "N(run_dataloaders.run_dataloader): %d, "
            "N(run_dataloaders.val_dataloder): %d, "
            "N(run_dataloaders.pool_dataloder): %d"
        ),
        linevul_config.method,
        sum(len(batch[0]) for batch in run_dataloaders.run_dataloader),
        sum(len(batch[0]) for batch in run_dataloaders.val_dataloader),
        sum(len(batch[0]) for batch in run_dataloaders.pool_dataloader),
    )

    (
        entropy_epistermic,
        entropy_aleatoric,
    ) = compute_data_pool_uq_metrics(config, ensemble, run_dataloaders)
    logger.info(
        (
            "computed uq for  %s at %d with len(train_dataset): %d, len(run_dataset): %d, "
            "len(val_dataset): %d, len(pool_dataset): %d"
        ),
        linevul_config.data_select_method,
        linevul_config.data_select_ratio,
        len(run_datasets.train_dataset),
        len(run_datasets.run_dataset),
        len(run_datasets.val_dataset),
        len(run_datasets.pool_dataset),
    )

    selection_size = round(len(linevul_dataset) * linevul_config.data_select_ratio)
    logger.info(
        "run %s method to select %d with model index %d",
        linevul_config.data_select_method,
        selection_size,
        linevul_config.model_index,
    )
    update_run_dataset(
        linevul_config.data_select_method,
        run_datasets,
        selection_size,
        linevul_config.model_index,
        entropy_epistermic,
        entropy_aleatoric,
    )
    return run_datasets.run_dataset


def save_selection_result(result_dataset, linevul_config):
    dir_path = os.path.dirname(linevul_config.output_file)
    os.makedirs(dir_path, exist_ok=True)
    torch.save(result_dataset, linevul_config.output_file)
    logger.info("saved select data to %s", linevul_config.output_file)


if __name__ == "__main__":
    init_logging(__file__, append=False)

    exp_config, linevul_exp_config = setup_experiment()

    selected_dataset = run_data_select_experiment(exp_config, linevul_exp_config)
    save_selection_result(selected_dataset, linevul_exp_config)
    print("done")
