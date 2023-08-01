import logging
import os
import sys
import torch

from uqmodel.stochasticensemble.logging_utils import init_logging
from uqmodel.stochasticensemble.train_utils import (
    EnsembleCheckpoint,
    SamplingFeatureDataSet,
    build_datasets,
    build_shifted_datasets,
    build_dataloaders,
    get_trained_model,
)
from uqmodel.stochasticensemble.eval_utils import (
    compute_uq_eval_metrics,
    result_dict_to_json,
)
from uqmodel.stochasticensemble.dataloader_utils import get_test_label
from uqmodel.stochasticensemble.experiment_config import (
    get_experiment_config,
    setup_reproduce,
)


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    init_logging(logger, __file__, append=False)

    config = get_experiment_config()

    if not os.path.exists(config.data.data_dir):
        raise ValueError(f"data_dir {config.data.data_dir} inaccessible")
        sys.exit(1)

    if config.reproduce:
        setup_reproduce(config.reproduce)

    if config.trainer.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = EnsembleCheckpoint(
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
    )

    train_dataset, val_dataset, test_dataset, ps_columns = build_datasets(config, ckpt)
    train_len = len(train_dataset)
    if config.data.sample_from_train and config.data.sample_from_train < 1.0:
        train_dataset = SamplingFeatureDataSet(
            train_dataset, config.data.sample_from_train
        )
        logger.info(
            "training dataset sampled with ratio {} results in from {} to {}".format(
                config.data.sample_from_train, train_len, len(train_dataset)
            )
        )
    train_dataset, val_dataset, test_dataset = build_shifted_datasets(
        config, train_dataset, val_dataset, test_dataset
    )
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        config, train_dataset, val_dataset, test_dataset
    )
    ensemble = get_trained_model(
        config,
        "disentangle",
        train_dataloader,
        val_dataloader,
        ps_columns,
        ckpt,
        criteria=None,
        device=device,
    )

    test_proba_pred_mean = list(
        ensemble.predict_mean_proba(
            test_dataloader, config.trainer.aleatoric_samples, device=device
        )
    )
    test_label_pred_list = []
    for _, labels in ensemble.compute_class_with_conf(test_proba_pred_mean):
        test_label_pred_list.append(labels)

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
    print(result_dict_to_json(result_dict))
