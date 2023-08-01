"""
ensemble UQ based on pre-trained ensemble model and saved checkpoints.
"""

import logging
import os
import torch

from uqmodel.ensemble.logging_utils import init_logging
from uqmodel.ensemble.dataloader_utils import get_test_label
from uqmodel.ensemble.eval_utils import (
    load_from_checkpoint_with_datashift,
    compute_uq_eval_metrics,
    result_dict_to_json,
)
from uqmodel.ensemble.experiment_config import get_experiment_config, setup_reproduce
from uqmodel.ensemble.eval_utils import EnsembleDisentangledUq


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    init_logging(logger, __file__)

    config = get_experiment_config()

    if config.reproduce:
        setup_reproduce(config.reproduce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        ensemble,
    ) = load_from_checkpoint_with_datashift("disentangle", config, device=device)

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
    print(result_dict_to_json(result_dict))
