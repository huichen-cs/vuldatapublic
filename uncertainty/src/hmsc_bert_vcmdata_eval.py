import json
import os
import torch

from uqmodel.bert.data import BertExperimentDatasets
from uqmodel.bert.experiment import (
    ExperimentConfig,
    get_experiment_config,
    setup_reproduce,
)
from uqmodel.bert.logging_utils import init_logging
from uqmodel.bert.eval_utils import EnsembleBertClassifierEvalautor
from hmsc_bert_vcmdata_train import get_trained_ensemble_model


def setup_experiment() -> ExperimentConfig:
    config = get_experiment_config()

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

    config.trainer.use_data = "use_checkpoint"

    return config


def main():
    init_logging(__file__, append=True)

    config = setup_experiment()

    datasets = BertExperimentDatasets(config, tag=None)
    ensemble = get_trained_ensemble_model(config, datasets, load_trained=True)
    evaluator = EnsembleBertClassifierEvalautor(config, ensemble, datasets.test_dataset)
    result_dict = evaluator.compute_eval_metrics(config.device)
    print(json.dumps(result_dict, indent=2))
    print("Eval completed")


if __name__ == "__main__":
    main()
