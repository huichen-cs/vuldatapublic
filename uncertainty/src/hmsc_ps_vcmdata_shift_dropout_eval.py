"""
dropout UQ based on pre-trained ensemble model with dropouts and saved checkpoints.
"""

import logging
import os
import torch

from uqmodel.ensemble.logging_utils import init_logging
from uqmodel.ensemble.dataloader_utils import get_test_label
from uqmodel.ensemble.dropout_mlc import DropoutClassifier
from uqmodel.ensemble.eval_utils import (
    DropoutUq,
    add_predictive_uq_to_result_dict,
    load_from_checkpoint_with_datashift,
    compute_uq_eval_metrics,
    result_dict_to_json
)
from uqmodel.ensemble.experiment_config import (
    get_extended_argparser,
    get_single_model_selection_criteria,
    get_experiment_config,
    setup_reproduce
)


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    init_logging(logger, __file__)

    parser = get_extended_argparser()
    selection_criteria = get_single_model_selection_criteria(parser)
    config = get_experiment_config(parser)

    if config.reproduce:
        setup_reproduce(config.reproduce)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader, test_dataloader, ensemble = load_from_checkpoint_with_datashift('predictive', config, device=device)
    dropout_model = DropoutClassifier(ensemble.select_model(val_dataloader, selection_critieria=selection_criteria, device=device))

    test_label_pred_list = []
    test_proba_samples_mean_list = []
    # predicted_class_samples_proba_list = []
    for results in dropout_model.predict_from_dropouts(test_dataloader, device=device):
        test_label_pred, test_proba, _, samples_proba, test_label_samples_pred, _, test_proba_samples_mean = results
        test_label_pred_list.append(test_label_samples_pred)
        test_proba_samples_mean_list.append(test_proba_samples_mean)
        # predicted_class_samples_proba_list.append(predicted_class_samples_proba)

    test_label_pred_tensor = torch.cat(test_label_pred_list).to(device)
    test_label_tensor = get_test_label(test_dataloader, device=device)
    # conf = torch.cat(predicted_class_samples_proba_list, dim=0)

    predicted_proba_tensor = torch.cat(test_proba_samples_mean_list, dim=0)
    # result_dict = compute_uq_eval_metrics(config, predicted_proba_tensor, conf, test_label_pred_tensor, test_label_tensor, py_script=__file__)
    result_dict = compute_uq_eval_metrics(config,
                                          predicted_proba_tensor,
                                        #   conf,
                                          test_label_pred_tensor,
                                          test_label_tensor,
                                          py_script=os.path.basename(__file__),
                                          metrics_list=['acc', 'precision', 'recall', 'f1', 'mcc', 'auprc', 'auroc'])
    result_dict = add_predictive_uq_to_result_dict(result_dict, DropoutUq(dropout_model, test_dataloader, n_stochastic_passes=1000))
    print(result_dict_to_json(result_dict))
