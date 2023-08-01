import json
import logging
import os

import torch

from uqmodel.shiftstochasticbert.data import BertExperimentDatasets
from uqmodel.shiftstochasticbert.experiment import (
    ExperimentConfig,
    get_experiment_config,
    setup_reproduce,
)
from uqmodel.shiftstochasticbert.logging_utils import init_logging
from uqmodel.shiftstochasticbert.eval_utils import (
    StochasticEnsembleBertClassifierEvalautor,
)
from htsc_bert_vcmdata_shift_train import get_trained_ensemble_model


logger = logging.getLogger(__name__)


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
    config.trainer.use_model = "use_checkpoint"

    return config


def main():
    init_logging(__file__, append=True)

    config = setup_experiment()

    datasets = BertExperimentDatasets(config, tag=None, dataset_name="VCMDATA")
    ensemble = get_trained_ensemble_model(config, datasets, load_trained=True)
    evaluator = StochasticEnsembleBertClassifierEvalautor(
        config, ensemble, datasets.test_dataset
    )
    result_dict = evaluator.compute_eval_metrics(config)
    print(json.dumps(result_dict, indent=2))
    logger.info("Eval completed")


if __name__ == "__main__":
    main()
