import argparse
import numpy as np
import os
import random
import torch


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-c config_file] [...]", description="Run an UQ experiment."
    )

    parser.add_argument("-c", "--config")

    return parser


class ExperimentConfig(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = os.path.expanduser("~/.hfcache")
        self.data_dir = "methods/VCMatch/data/SAP"
        self.batch_size = 10
        self.max_encoding_len = 512
        self.num_classes = 2
        self.shuffle_data = True
        self._train_ratio = 0.9
        self._test_ratio = 0.1
        self.train_test_ratios = [self._train_ratio, self._test_ratio]
        self.val_ratio = 0.1
        self.imbalance_ratio = 1
        #
        self.model_ensemble_size = 5
        self.dropout_proba = 0.1
        #
        self.trainer_max_dataloader_workers = 0
        self.trainer_cpu_only = False
        self.trainer_use_model = "try_checkpoint"
        self.trainer_early_stopping_patience = 5
        self.trainer_early_stopping_min_delta = 0
        self.trainer_lr_scheduler_step_size = 10
        self.trainer_lr_scheduler_gamma = 0.1
        self.trainer_max_iter = 15
        self.trainer_optimizer_init_lr = 2e-5
        self.trainer_aleatoric_samples = 100
        #
        self.trainer_checkpoint_dir_path = "uq_testdata_ckpt/activede/bert/sap/v1/test1"
        self.trainer_checkpoint_warmup_epochs = 1
        #
        self.reproduce = True
        self.torch_manual_seed = 3117
        self.py_random_seed = 1115
        self.np_random_seed = 7310
        #
        self.pin_memory = False
        #
        self.trainer_tensorboard_logdir = "tb_logdir_bert"
        #
        self.config_fn = ""
        #
        self.action = None
        #
        self.num_workers = None


def setup_reproduce(config):
    """
    https://pytorch.org/docs/stable/notes/randomness.html.
    """
    if config.torch_manual_seed:
        torch.manual_seed(config.torch_manual_seed)
    if config.py_random_seed:
        random.seed(config.py_random_seed)
    if config.np_random_seed:
        np.random.seed(config.np_random_seed)
