"""
ensemble UQ based on pre-trained ensemble model and saved checkpoints.
"""

import logging
import os
import torch

from uqmodel.stochasticensemble.logging_utils import init_logging
from uqmodel.stochasticensemble.dataloader_utils import get_test_label
from uqmodel.stochasticensemble.eval_utils import (
    load_from_checkpoint_with_datashift,
    compute_uq_eval_metrics,
    result_dict_to_json,
)
from uqmodel.stochasticensemble.experiment_config import (
    get_extended_argparser,
    get_experiment_config,
    get_single_model_selection_criteria,
    setup_reproduce,
)
from uqmodel.stochasticensemble.eval_utils import (
    StochasticEnsembleModelSelector,
)
from uqmodel.stochasticensemble.stochastic_vanilla_mlc import (
    StochasticVanillaClassifier,
)


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    init_logging(logger, __file__)

    parser = get_extended_argparser()
    selection_criteria = get_single_model_selection_criteria(parser)
    config = get_experiment_config(parser)

    if config.reproduce:
        setup_reproduce(config.reproduce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        ensemble,
    ) = load_from_checkpoint_with_datashift("disentangle", config, device=device)

    member_selector = StochasticEnsembleModelSelector(ensemble, val_dataloader, config)

    logger.info(
        "select ensemble member model via criteria {}".format(selection_criteria)
    )
    vanilla_model = StochasticVanillaClassifier(
        member_selector.select_member_model(selection_critieria=selection_criteria)
    )

    test_proba_pred_mean = list(
        vanilla_model.generate_mean_proba(
            test_dataloader, config.trainer.aleatoric_samples, device=device
        )
    )

    test_conf_pred_list, test_label_pred_list = [], []
    for confs, labels in vanilla_model.compute_class_with_conf(test_proba_pred_mean):
        test_conf_pred_list.append(confs)
        test_label_pred_list.append(labels)

    test_conf_pred_tensor = torch.cat(test_conf_pred_list, dim=0).to(device)
    labels_pred = torch.cat(test_label_pred_list, dim=0).to(device)
    targets = get_test_label(test_dataloader, device=device)

    probas_pred = torch.cat(test_proba_pred_mean, dim=0)
    result_dict = compute_uq_eval_metrics(
        config,
        probas_pred,
        labels_pred,
        targets,
        py_script=os.path.basename(__file__),
        metrics_list=["acc", "precision", "recall", "f1", "mcc", "auprc", "auroc"],
    )

    uq_list = []
    for idx in range(len(targets)):
        target = targets[idx].item()
        label_pred = labels_pred[idx].item()
        label_conf = test_conf_pred_tensor[idx].item()
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
            "quadrant": quadrant,
        }
        uq_list.append(uq_dict)
    result_dict["uq"] = uq_list
    print(result_dict_to_json(result_dict))
