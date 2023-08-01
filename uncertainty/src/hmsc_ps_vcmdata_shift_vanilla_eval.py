"""
vanilla UQ based on pre-trained ensemble model and saved checkpoints.
"""

import logging
import os
import torch

from uqmodel.ensemble.logging_utils import init_logging
from uqmodel.ensemble.vanilla_mlc import VanillaClassifier
from uqmodel.ensemble.dataloader_utils import get_test_label
from uqmodel.ensemble.eval_utils import (
    add_predictive_uq_to_result_dict,
    VanillaUq,
    load_from_checkpoint_with_datashift,
    compute_uq_eval_metrics,
    result_dict_to_json,
)
from uqmodel.ensemble.experiment_config import (
    get_extended_argparser,
    get_single_model_selection_criteria,
    get_experiment_config,
    setup_reproduce,
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
    ) = load_from_checkpoint_with_datashift("predictive", config, device=device)
    vallina_model = VanillaClassifier(
        ensemble.select_model(
            val_dataloader, selection_critieria=selection_criteria, device=device
        )
    )

    test_proba_pred = list(vallina_model.predict_proba(test_dataloader, device=device))
    test_label_pred_list = []
    # test_conf_pred_list = []
    for _, label_bacth in vallina_model.compute_class_with_conf(
        test_proba_pred, device=device
    ):
        test_label_pred_list.append(label_bacth)
        # test_conf_pred_list.append(conf_batch)

    test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).to(device)
    # test_conf_pred_tensor = torch.cat(test_conf_pred_list, dim=0).to(device)
    test_label_tensor = get_test_label(test_dataloader, device=device)

    predicted_proba_tensor = torch.cat(test_proba_pred, dim=0)
    # result_dict = compute_uq_eval_metrics(config, predicted_proba_tensor, test_conf_pred_tensor, test_label_pred_tensor, test_label_tensor, py_script=__file__)
    result_dict = compute_uq_eval_metrics(
        config,
        predicted_proba_tensor,
        #   test_conf_pred_tensor,
        test_label_pred_tensor,
        test_label_tensor,
        py_script=os.path.basename(__file__),
        metrics_list=["acc", "precision", "recall", "f1", "mcc", "auprc", "auroc"],
    )
    result_dict = add_predictive_uq_to_result_dict(
        result_dict, VanillaUq(vallina_model, test_dataloader)
    )
    print(result_dict_to_json(result_dict))
