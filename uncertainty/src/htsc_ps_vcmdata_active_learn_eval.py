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
import numpy as np
import os
import torch
from uqmodel.stochasticensemble.logging_utils import init_logging
from uqmodel.stochasticensemble.experiment_config import ExperimentConfig
from uqmodel.stochasticensemble.dataloader_utils import get_test_label
from uqmodel.stochasticensemble.active_learn import EnsembleCheckpoint
from uqmodel.stochasticensemble.eval_utils import (
    EnsembleDisentangledUq,
    compute_uq_eval_metrics,
    result_dict_to_json,
)
from uqmodel.stochasticensemble.ensemble_mlc import StochasticEnsembleClassifier
from uqmodel.stochasticensemble.stochastic_mlc import StochasticMultiLayerClassifier
from uqmodel.stochasticensemble.ensemble_trainer import EnsembleTrainer
from uqmodel.stochasticensemble.experiment_config import (
    get_experiment_config,
    init_argparse,
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
    config.output_log_sigma = False

    config = get_extended_args(config, parser)

    config.trainer.use_model = 'use_checkpoint'

    return config


def load_from_checkpoint(config: ExperimentConfig, tag=None):
    ckpt = EnsembleCheckpoint(
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
        tag=tag,
        train=False,
    )
    try:
        _, _, test_dataset, ps_columns = ckpt.load_datasets()
        logger.info(
            "loaded train/val/test datasets from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        )
    except FileNotFoundError as err:
        logger.error(
            "failed to load train/va/test datasets from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        )
        raise err

    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                 batch_size=config.trainer.batch_size,
    #                                                 num_workers=config.num_workers,
    #                                                 pin_memory=False)
    # val_dataloader =  torch.utils.data.DataLoader(val_dataset,
    #                                                 batch_size=config.trainer.batch_size,
    #                                                 num_workers=config.num_workers,
    #                                                 pin_memory=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    ensemble = StochasticEnsembleClassifier(
        StochasticMultiLayerClassifier,
        config.output_log_sigma,  # log_sigma <- False
        config.model.ensemble_size,
        len(ps_columns),
        2,
        neurons=config.model.num_neurons,
        dropouts=config.model.dropout_ratios,
        activation=torch.nn.LeakyReLU(),
    )
    trainer = EnsembleTrainer(
        ensemble,
        criteria=None,
        lr_scheduler=None,
        max_iter=config.trainer.max_iter,
        init_lr=config.trainer.optimizer.init_lr,
        device=config.device,
        checkpoint=ckpt,
        earlystopping=None,
        n_samples=config.trainer.aleatoric_samples,
        ouput_log_sigma=config.output_log_sigma,
    )

    try:
        ensemble = trainer.load_checkpoint()
        logger.info("use the ensemble model from checkpoint, no training")
    except FileNotFoundError as err:
        raise ValueError(
            "failed to load pre-trained ensemble model from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        ) from err
    return test_dataloader, ps_columns, ensemble


def load_ensemble_from_checkpoint(config: ExperimentConfig, ps_columns: list, tag=None):
    ckpt = EnsembleCheckpoint(
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
        tag=tag,
        train=False,
    )
    ensemble = StochasticEnsembleClassifier(
        StochasticMultiLayerClassifier,
        config.output_log_sigma,  # log_sigma <- False
        config.model.ensemble_size,
        len(ps_columns),
        2,
        neurons=config.model.num_neurons,
        dropouts=config.model.dropout_ratios,
        activation=torch.nn.LeakyReLU(),
    )
    trainer = EnsembleTrainer(
        ensemble,
        criteria=None,
        lr_scheduler=None,
        max_iter=config.trainer.max_iter,
        init_lr=config.trainer.optimizer.init_lr,
        device=config.device,
        checkpoint=ckpt,
        earlystopping=None,
        n_samples=config.trainer.aleatoric_samples,
        ouput_log_sigma=config.output_log_sigma,
    )

    try:
        ensemble = trainer.load_checkpoint()
        logger.info("use the ensemble model from checkpoint, no training")
    except FileNotFoundError as err:
        raise ValueError(
            "failed to load pre-trained ensemble model from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        ) from err
    return ensemble


def compute_eval_metrics(ensemble, test_dataloader, device=None):
    test_proba_pred_mean = list(
        ensemble.predict_mean_proba(
            test_dataloader, config.trainer.aleatoric_samples, device=device
        )
    )
    test_conf_pred_list, test_label_pred_list = [], []
    for confs, labels in ensemble.compute_class_with_conf(test_proba_pred_mean):
        test_conf_pred_list.append(confs)
        test_label_pred_list.append(labels)

    test_conf_pred_tensor = torch.cat(test_conf_pred_list, dim=0).to(device)
    test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).to(device)
    test_label_tensor = get_test_label(test_dataloader, device=device)

    predicted_proba_tensor = torch.cat(test_proba_pred_mean, dim=0)
    result_dict = compute_uq_eval_metrics(
        config,
        predicted_proba_tensor,
        test_label_pred_tensor,
        test_label_tensor,
        py_script=os.path.basename(__file__),
        metrics_list=["acc", "precision", "recall", "f1", "mcc", "auprc", "auroc"],
    )

    uq_list = []
    uq = EnsembleDisentangledUq(
        ensemble, test_dataloader, config.trainer.aleatoric_samples, device=device
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
    for idx in range(len(test_label_tensor)):
        target = test_label_tensor[idx].item()
        label_pred = test_label_pred_tensor[idx].item()
        label_conf = test_conf_pred_tensor[idx].item()
        proba_ale = proba_mean_aleatoric[idx].detach().cpu().numpy().tolist()
        proba_ale_std = proba_std_aleatoric[idx].item()
        proba_epi = proba_mean_epistermic[idx].detach().cpu().numpy().tolist()
        proba_epi_std = proba_std_epistermic[idx].item()
        entropy_ale = entropy_aleatoric[idx].item()
        entropy_epi = entropy_epistermic[idx].item()
        proba_std_instance = proba_std[idx].item()
        entropy_instance = entropy_all[idx].item()
        muinfo_instance = muinfo_all[idx].item()
        logits_mu_mean = mu_mean[idx].detach().cpu().numpy().tolist()
        logits_sigma_aleatoric = sigma_aleatoric[idx].detach().cpu().numpy().tolist()
        logits_sigma_epistermic = sigma_epistermic[idx].detach().cpu().numpy().tolist()
        if target == label_pred and target == 1:
            quadrant = "TP"
        elif target == label_pred and target == 0:
            quadrant = "TN"
        elif target != label_pred and label_pred == 1:
            quadrant = "FP"
        else:
            quadrant = "FN"
        uq_dict = {
            "index": idx,
            "target": target,
            "label_pred": label_pred,
            "label_conf": label_conf,
            "proba_ale": proba_ale,
            "proba_ale_std": proba_ale_std,
            "entropy_ale": entropy_ale,
            "proba_epi": proba_epi,
            "proba_epi_std": proba_epi_std,
            "entropy_epi": entropy_epi,
            "proba_std": proba_std_instance,
            "entropy": entropy_instance,
            "muinfo": muinfo_instance,
            "mu_mean": logits_mu_mean,
            "sigma_aleatoric": logits_sigma_aleatoric,
            "sigma_epistermic": logits_sigma_epistermic,
            "quadrant": quadrant,
        }
        uq_list.append(uq_dict)
    result_dict["uq"] = uq_list
    return result_dict


def prepare_for_json_dump(result_dict):
    for k in result_dict.keys():
        result_dict[k] = (
            result_dict[k].cpu().numpy()
            if isinstance(result_dict[k], torch.Tensor)
            else result_dict[k]
        )
        result_dict[k] = (
            result_dict[k].tolist()
            if isinstance(result_dict[k], np.ndarray)
            else result_dict[k]
        )
    return result_dict


def result_to_json(result_dict):
    for k in result_dict:
        result = result_dict[k]
        if isinstance(result, list):
            updated_list = []
            for r in result:
                r = prepare_for_json_dump(r)
                updated_list.append(r)
            result_dict[k] = updated_list
        elif isinstance(result, dict):
            result = prepare_for_json_dump(result)
            result_dict[k] = result
        else:
            raise ValueError("unsupported data type {}".format(type(result)))
    json = result_dict_to_json(result_dict)
    return json


def run_experiment(config: ExperimentConfig) -> dict:
    result_dict = dict()
    for method in [
        "init",
        "ehal",
        "elah",
        "ehah",
        "elal",
        "aleh",
        "ahel",
        "aheh",
        "alel",
    ]:
        result_dict[method] = []

    test_dataloader, ps_columns, ensemble = load_from_checkpoint(config, tag=None)
    result_dict["init"] = compute_eval_metrics(
        ensemble, test_dataloader, device=config.device
    )

    for method in ["ehal", "elah", "ehah", "elal", "aleh", "ahel", "aheh", "alel"]:
        if method == config.action or config.action == "all":
            result_list = []
            for i in range(5):
                logger.info("run {} method for step {}".format(method, i))
                ensemble = load_ensemble_from_checkpoint(
                    config, ps_columns, tag="{}_{}".format(i, method)
                )

                result_list.append(
                    compute_eval_metrics(ensemble, test_dataloader, device=config.device)
                )
                logger.info("done {} at {}".format(method, i))
            result_dict[method] = result_list

    return result_dict


if __name__ == "__main__":
    init_logging(logger, __file__, append=True)

    config = setup_experiment()

    result_dict = run_experiment(config)
    json = result_to_json(result_dict)
    print(json)
