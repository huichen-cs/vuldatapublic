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
import gc
import logging
import numpy as np
import os
import torch.multiprocessing as mp
import torch
from copy import deepcopy
from datetime import datetime
from typing import Union
from uqmodel.bert.checkpoint import EnsembleCheckpoint
from uqmodel.stochasticbert.logging_utils import init_logging, get_global_logfilename
from uqmodel.stochasticbert.data import (
    BertExperimentDatasets,
    BertExperimentDataLoaders,
)
from uqmodel.stochasticbert.dataloader_utils import get_test_label
from uqmodel.stochasticbert.ensemble_bert import StochasticEnsembleBertClassifier
from uqmodel.stochasticbert.experiment import (
    ExperimentConfig,
    init_argparse,
    setup_reproduce,
)
from uqmodel.stochasticbert.eval_utils import (
    EnsembleDisentangledUq,
    result_dict_to_json,
    compute_uq_eval_metrics,
)
from htsc_bert_vcmdata_active_learn_train_ext import get_trained_ensemble_model


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
        "-a",
        "--action",
        help="active learning action in {}".format(
            [
                "all",
                "init",
            ]
            + get_active_learning_method_list()
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
    else:
        config.action = "all"
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

    config.trainer.use_model = "use_checkpoint"
    return config


def get_datetime_jobid():
    date_time = datetime.now().strftime("%Y%m%d%H%M%S")
    return date_time


def compute_eval_metrics(
    ensemble: StochasticEnsembleBertClassifier,
    test_dataloader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
):
    test_proba_pred_mean = list(
        [
            p.cpu()
            for p in ensemble.predict_mean_proba(
                test_dataloader, config.trainer.aleatoric_samples, device=config.device
            )
        ]
    )
    test_conf_pred_list, test_label_pred_list = [], []
    for confs, labels in ensemble.compute_class_with_conf(test_proba_pred_mean):
        test_conf_pred_list.append(confs)
        test_label_pred_list.append(labels)

    test_conf_pred_tensor = torch.cat(test_conf_pred_list, dim=0).cpu()
    test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).cpu()
    test_label_tensor = get_test_label(test_dataloader, device=config.device).cpu()

    predicted_proba_tensor = torch.cat(test_proba_pred_mean, dim=0).cpu()
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
        ensemble,
        test_dataloader,
        config.trainer.aleatoric_samples,
        device=config.device,
        mp=True,
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
        if not mu_mean[idx].requires_grad:
            logits_mu_mean = mu_mean[idx].numpy().tolist()
        else:
            logits_mu_mean = mu_mean[idx].detach().cpu().numpy().tolist()
        if not sigma_aleatoric[idx].requires_grad:
            logits_sigma_aleatoric = sigma_aleatoric[idx].numpy().tolist()
        else:
            logits_sigma_aleatoric = (
                sigma_aleatoric[idx].detach().cpu().numpy().tolist()
            )
        if not sigma_epistermic[idx].requires_grad:
            logits_sigma_epistermic = sigma_epistermic[idx].numpy().tolist()
        else:
            logits_sigma_epistermic = (
                sigma_epistermic[idx].detach().cpu().numpy().tolist()
            )

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
            if "_data_increment" in k:
                result_dict[k] = result
                continue
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


def _mp_run_one_evaluation(config, run_datasets, queue, logfilename):
    init_logging(logfilename, append=True)
    run_dataloaders = BertExperimentDataLoaders(config, run_datasets, train=False)
    try:
        ensemble = get_trained_ensemble_model(config, run_datasets, load_trained=True)
        eval_metrics = compute_eval_metrics(
            ensemble, run_dataloaders.test_dataloader, config
        )
        queue.put(eval_metrics)
    except Exception as err:
        logger.info("load ensemble model: " + repr(err))
        queue.put(None)

 
def run_one_evaluation(config, run_datasets):
    print_allocation("enter run_one_evaluation", print)

    # mp.set_sharing_strategy('file_system')
    mp.set_sharing_strategy("file_descriptor")
    mp.set_start_method(method="forkserver", force=True)
    manager = mp.Manager()
    queue = manager.Queue()
    # ctx = mp.get_context('spawn')
    ctx = mp.get_context("forkserver")
    p = ctx.Process(
        target=_mp_run_one_evaluation,
        args=(config, run_datasets, queue, get_global_logfilename()),
    )
    p.start()
    eval_metrics = queue.get(block=True, timeout=None)
    p.join()

    print_allocation("leave run_one_evaluation, after cuda.synchronize()", print)
    return eval_metrics


def _run_one_evaluation(config, run_datasets):
    print_allocation("enter run_one_evaluation", print)
    run_dataloaders = BertExperimentDataLoaders(config, run_datasets, train=False)
    print_allocation("in run_one_evaluation, got data loaders", print)
    ensemble = get_trained_ensemble_model(config, run_datasets, load_trained=True)
    print_allocation("in run_one_evaluation, got model", print)
    eval_metrics = compute_eval_metrics(
        ensemble, run_dataloaders.test_dataloader, device=config.device
    )
    print_allocation("in run_one_evaluation, got eval metrics", print)
    if ensemble:
        for model in ensemble:
            model.cpu()
        del ensemble
    if run_dataloaders:
        del run_dataloaders
    print_allocation("in run_one_evaluation, before cuda.synchronize()", print)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    print_allocation("leave run_one_evaluation, after cuda.synchronize()", print)
    return eval_metrics


def print_allocation(title, disp):
    bytes = torch.cuda.memory_allocated(torch.device("cuda"))
    disp("{}: CUDA memory allocated: {} bytes".format(title, bytes))


def debug_cuda_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print("{:<20}".format(str(obj)), type(obj), obj.size())
        except Exception:
            pass


def run_eval_experiment(config: ExperimentConfig) -> dict:
    n_data_parts = 10
    print_allocation("ener run_eval_experiment", print)
    assert config.trainer.use_model == "use_checkpoint"

    result_dict = dict()
    for method in [
        "init",
    ] + get_active_learning_method_list():
        result_dict[method] = []

    experiment_datasets = BertExperimentDatasets(config, None, dataset_name="VCMDATA")
    eval_metrics = run_one_evaluation(config, experiment_datasets)
    debug_cuda_memory()
    # experiment_dataloaders = BertExperimentDataLoaders(config, experiment_datasets, train=False)
    # ensemble = get_trained_ensemble_model(config, experiment_datasets, load_trained=True)
    # result_dict['init'] = compute_eval_metrics(ensemble,
    #                                         experiment_dataloaders.test_dataloader,
    #                                         device=config.device)
    result_dict["init"] = eval_metrics
    # del experiment_dataloaders
    # gc.collect()
    # torch.cuda.empty_cache()

    for method in get_active_learning_method_list():
        if method == config.action or config.action == "all":
            result_list = []
            data_increment_list = []

            run_datasets = deepcopy(experiment_datasets)
            # del experiment_datasets
            # gc.collect()
            # torch.cuda.empty_cache()

            logger.info(
                "begin {} with len(run_dataset): {}, len(pool_dataset): {}".format(
                    method,
                    len(run_datasets.run_dataset),
                    len(run_datasets.pool_dataset),
                )
            )
            for i in range(n_data_parts):
                try:
                    logger.info("run {} method for step {}".format(method, i))
                    run_datasets.update_checkpoint("{}_{}".format(i, method))
                    eval_metrics = run_one_evaluation(config, run_datasets)
                    if not eval_metrics:
                        logger.info("cannot evaluate for method " + method)
                        raise FileNotFoundError("model for method " + method + " not found")
                    # run_dataloaders = BertExperimentDataLoaders(config, run_datasets, train=False)

                    # ensemble = get_trained_ensemble_model(config, run_datasets, load_trained=True)
                    # eval_metrics = compute_eval_metrics(ensemble, run_dataloaders.test_dataloader, device=config.device)
                    result_list.append(eval_metrics)
                    data_increment_list.append(i)
                    logger.info("done {} at {}".format(method, i))

                    # del run_dataloader, ensemble
                    # gc.collect()
                    # torch.cuda.empty_cache()
                except Exception as err:
                    logger.info(
                        "result for {} at {} does not exist, due to {}".format(
                            method, i, err
                        )
                    )
            result_dict[method] = result_list
            result_dict["{}_data_increment".format(method)] = data_increment_list
    return result_dict


if __name__ == "__main__":
    # print("begin")
    init_logging(__file__, append=False)

    config = setup_experiment()

    result_dict = run_eval_experiment(config)
    json = result_to_json(result_dict)
    print(json)
    # print("done")
