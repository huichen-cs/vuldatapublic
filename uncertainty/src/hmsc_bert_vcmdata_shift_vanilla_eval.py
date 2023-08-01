import json
import logging
import os
import torch
from typing import Tuple
from uqmodel.shiftbert.data import BertExperimentDatasets
from uqmodel.shiftbert.experiment import (
    ExperimentConfig,
    get_experiment_config,
    get_extended_argparser,
    get_single_model_selection_criteria,
    setup_reproduce,
)
from uqmodel.shiftbert.logging_utils import init_logging
from uqmodel.shiftbert.eval_utils import (
    VanillaBertClassifierEvalautor,
    EnsembleModelSelector,
)
from hmsc_bert_vcmdata_shift_train import get_trained_ensemble_model


logger = logging.getLogger(__name__)


def setup_experiment() -> Tuple[ExperimentConfig, str]:
    parser = get_extended_argparser()
    selection_criteria = get_single_model_selection_criteria(parser)
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

    config.trainer.use_data = "use_checkpoint"
    config.trainer.use_model = "use_checkpoint"

    return config, selection_criteria


def main():
    init_logging(__file__, append=True)

    config, selection_criteria = setup_experiment()
    logger.info(
        "Evaluate dropout UQ model for ensemble member selected via {}".format(
            selection_criteria
        )
    )

    datasets = BertExperimentDatasets(config, tag=None, dataset_name="PSDATA")
    ensemble = get_trained_ensemble_model(config, datasets, load_trained=True)
    member_selector = EnsembleModelSelector(ensemble, datasets.val_dataset, config)
    vanilla_model = member_selector.select_member_model(
        selection_critieria=selection_criteria
    )
    torch.cuda.empty_cache()  # only one member model needed on CUDA
    evaluator = VanillaBertClassifierEvalautor(
        config, vanilla_model, datasets.test_dataset
    )
    result_dict = evaluator.compute_eval_metrics(config.device)
    print(json.dumps(result_dict, indent=2))
    logger.info("Eval completed")


if __name__ == "__main__":
    main()
