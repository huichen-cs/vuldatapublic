"""
vanilla UQ based on pre-trained ensemble model and saved checkpoints.
"""

import logging
import os
import torch

from uqmodel.ensemble.logging_utils import init_logging
from uqmodel.ensemble.dataloader_utils import get_test_label
from uqmodel.ensemble.eval_utils import (
    EnsembleUq, add_predictive_uq_to_result_dict,
    load_from_checkpoint_with_datashift, compute_uq_eval_metrics, result_dict_to_json
)
from uqmodel.ensemble.experiment_config import get_experiment_config, setup_reproduce


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    init_logging(logger, __file__)

    config = get_experiment_config()

    if config.reproduce:
        setup_reproduce(config.reproduce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader, ensemble = load_from_checkpoint_with_datashift('predictive', config, device=device)

    test_proba_pred_mean = list(ensemble.predict_mean_proba(test_dataloader, device=device))
    test_label_pred_list = []
    # test_conf_pred_list = []
    for _, labels in ensemble.compute_class_with_conf(test_proba_pred_mean, device=device):
        test_label_pred_list.append(labels)
        # test_conf_pred_list.append(confs)
        
    test_label_pred_tensor = torch.cat(test_label_pred_list, dim=0).to(device)
    # test_conf_pred_tensor = torch.cat(test_conf_pred_list, dim=0).to(device)
    test_label_tensor = get_test_label(test_dataloader, device=device)

    predicted_proba_tensor = torch.cat(test_proba_pred_mean, dim=0)
    # result_dict = compute_uq_eval_metrics(config, predicted_proba_tensor, test_conf_pred_tensor, test_label_pred_tensor, test_label_tensor, py_script=__file__)
    result_dict = compute_uq_eval_metrics(config,
                                          predicted_proba_tensor,
                                        #   test_conf_pred_tensor,
                                          test_label_pred_tensor,
                                          test_label_tensor,
                                          py_script=os.path.basename(__file__),
                                          metrics_list=['acc', 'precision', 'recall', 'f1', 'mcc', 'auprc', 'auroc'])
    result_dict = add_predictive_uq_to_result_dict(result_dict, EnsembleUq(ensemble, test_dataloader))
    print(result_dict_to_json(result_dict))