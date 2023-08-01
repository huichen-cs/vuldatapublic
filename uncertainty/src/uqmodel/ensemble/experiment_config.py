import argparse
import ast
import logging
import numpy as np
import os
import torch
import random
from configparser import ConfigParser, ExtendedInterpolation


logger = logging.getLogger("experiment_config")


def setup_reproduce(config):
    """
    According to https://pytorch.org/docs/stable/notes/randomness.html.
    """
    if config.torch_manual_seed:
        torch.manual_seed(config.torch_manual_seed)
    if config.py_random_seed:
        random.seed(config.py_random_seed)
    if config.np_random_seed:
        np.random.seed(config.np_random_seed)


def str_to_list(s):
    return ast.literal_eval(s)


def str_to_number(s):
    return ast.literal_eval(s)


def str_to_boolean(s):
    return ast.literal_eval(s)


class ExperimentConfig(object):
    def __init__(self, config_fn=None):
        self.__init_default()

        if config_fn:
            self.config_fn = config_fn
            config = ConfigParser(interpolation=ExtendedInterpolation())
            config.read(config_fn)

            self.__update_from_config(config)

    def __str__(self):
        return (
            "{{reproduce: {}, data: {}, datashift:{}, model: {}, trainer: {}}}".format(
                self.reproduce, self.data, self.datashift, self.model, self.trainer
            )
        )

    def __init_default(self):
        self.config_fn = None
        self.data = self.DataConfig()
        self.model = self.ModelConfig()
        self.trainer = self.TrainerConfig()
        self.datashift = self.DataShiftConfig()
        self.reproduce = self.ReproduceConfig()

    def __update_from_config(self, config):
        if "reproduce" in config:
            self.reproduce.update_from_config(config["reproduce"])
        if "data" in config:
            self.data.update_from_config(config["data"])
        if "model" in config:
            self.model.update_from_config(config["model"])
        if "trainer" in config:
            self.trainer.update_from_config(config)
        if "data.shift" in config:
            self.datashift.update_from_config(config["data.shift"])

    class ReproduceConfig(object):
        def __init__(self):
            self.torch_manual_seed = None
            self.py_random_seed = None
            self.np_random_seed = None
            self.data_sampling_seed = None

        def update_from_config(self, config):
            if "torch_manual_seed" in config:
                self.torch_manual_seed = str_to_number(config["torch_manual_seed"])
            if "py_random_seed" in config:
                self.py_random_seed = str_to_number(config["py_random_seed"])
            if "np_random_seed" in config:
                self.np_random_seed = str_to_number(config["np_random_seed"])
            if "data_sampling_seed" in config:
                self.data_sampling_seed = str_to_number(config["data_sampling_seed"])

        def __str__(self):
            return "{{torch_manual_seed: {}, py_random_seed: {}, np_random_seed: {}, data_sampling_seed:{}}}".format(
                self.torch_manual_seed,
                self.py_random_seed,
                self.np_random_seed,
                self.data_sampling_seed,
            )

    class DataConfig(object):
        def __init__(self):
            self.data_dir = "methods/VCMatch/data"
            self.imbalance_ratio = 1
            self.cve_sample_size = 1.0
            self.train_test_ratios = [0.8, 0.2]
            self.val_ratio = 0.2
            self.shift_data_portion = None
            self.sample_from_train = None

        def update_from_config(self, config):
            if "data_dir" in config:
                self.data_dir = config["data_dir"]
            if "imbalance_ratio" in config:
                self.imbalance_ratio = str_to_number(config["imbalance_ratio"])
            if "cve_sample_size" in config:
                self.cve_sample_size = str_to_number(config["cve_sample_size"])
            if "train_test_ratios" in config:
                self.train_test_ratios = str_to_list(config["train_test_ratios"])
            if "val_ratio" in config:
                self.val_ratio = str_to_number(config["val_ratio"])
            if "shift_data_portion" in config:
                self.shift_data_portion = str_to_number(config["shift_data_portion"])
            if "sample_from_train" in config:
                self.sample_from_train = str_to_number(config["sample_from_train"])

        def __str__(self):
            return "{{data_dir: {}, imbalance_ratio: {}, cve_sample_size: {}}}".format(
                self.data_dir, self.imbalance_ratio, self.cve_sample_size
            )

    class DataShiftConfig(object):
        def __init__(self):
            self.param_dict = None

        def update_from_config(self, config):
            if "type" in config:
                self.type = config["type"]
            else:
                raise ValueError("invalid configuration for datashfit, type expected")
            if self.type != "IndependentGaussianNoiseDataShift":
                raise ValueError("shift type {} not yet supported".format(self.type))
            if "mu" in config:
                self.mu = str_to_number(config["mu"])
            else:
                raise ValueError("invalid configuration for datashfit, mu expected")
            if "sigma" in config:
                self.sigma = str_to_number(config["sigma"])
            else:
                raise ValueError("invalid configuration for datashfit, sigma expected")
            self.param_dict = {
                "type": "IndependentGaussianNoiseDataShift",
                "mu": self.mu,
                "sigma": self.sigma,
            }

        def __str__(self):
            return "{}".format(self.param_dict)

    class ModelConfig(object):
        def __init__(self):
            self.ensemble_size = 5
            self.num_neurons = [1024, 2048, 512]
            self.dropout_ratios = [None, 0.25, 0.25]

        def update_from_config(self, config):
            if "ensemble_size" in config:
                self.ensemble_size = int(config["ensemble_size"])
            if "num_neurons" in config:
                self.num_neurons = str_to_list(config["num_neurons"])
            if "dropout_ratios" in config:
                self.dropout_ratios = str_to_list(config["dropout_ratios"])

        def __str__(self):
            return "{{ensemble_size: {}, num_neurons: {}, dropout_ratios: {}}}".format(
                self.ensemble_size, self.num_neurons, self.dropout_ratios
            )

    class TrainerConfig(object):
        _SPLIT_DATA = set(
            ["sanity_check", "train_val_test"]
        )  # how train/val/tesst data are used
        _USE_DATA = set(["try_checkpoint"])  # how models are used for training
        _USE_MODEL = set(["try_checkpoint", "from_pretrain", "from_scratch"])

        def __init__(self):
            self.split_data = "sanity_check"
            self.use_data = "try_checkpoint"
            self.use_model = "try_checkpoint"
            self.batch_size = 128
            self.max_dataloader_workers = 8
            self.max_iter = 1000
            self.early_stopping = self.EarlyStopping()
            self.checkpoint = self.CheckPointing()
            self.optimizer = self.Optimizer()
            self.criteria = self.Criteria()
            self.lr_scheduler = self.LearningRateScheduler()
            self.pin_memory = False
            self.cpu_only = False
            self.aleatoric_samples = 100
            self.tensorboard_logdir = None

        def update_from_config(self, config):
            if "split_data" in config["trainer"]:
                if config["trainer"]["split_data"] not in self._SPLIT_DATA:
                    raise ValueError(
                        "expected trainer:split_data to be in {}".format(
                            self._SPLIT_DATA
                        )
                    )
                self.split_data = config["trainer"]["split_data"]
            if "use_data" in config["trainer"]:
                if config["trainer"]["use_data"] not in self._USE_DATA:
                    raise ValueError(
                        "expected trainer:use_data to be in {}".format(self._USE_DATA)
                    )
                self.use_data = config["trainer"]["use_data"]
            if "use_model" in config["trainer"]:
                if config["trainer"]["use_model"] not in self._USE_MODEL:
                    raise ValueError(
                        "expected trainer:use_model to be in {}".format(self._USE_MODEL)
                    )
                self.use_model = config["trainer"]["use_model"]
            if "batch_size" in config["trainer"]:
                self.batch_size = int(config["trainer"]["batch_size"])
            if "max_iter" in config["trainer"]:
                self.max_iter = int(config["trainer"]["max_iter"])
            if "max_dataloader_workers" in config["trainer"]:
                self.max_dataloader_workers = str_to_boolean(
                    config["trainer"]["max_dataloader_workers"]
                )
            if "pin_memory" in config["trainer"]:
                self.pin_memory = str_to_boolean(config["trainer"]["pin_memory"])
            if "cpu_only" in config["trainer"]:
                self.pin_memory = str_to_boolean(config["trainer"]["cpu_only"])
            if "aleatoric_samples" in config["trainer"]:
                self.aleatoric_samples = str_to_number(
                    config["trainer"]["aleatoric_samples"]
                )
            if "tensorboard_logdir" in config["trainer"]:
                self.tensorboard_logdir = config["trainer"]["tensorboard_logdir"]
            if "trainer.earlystopping" in config:
                self.early_stopping.update_from_config(config["trainer.earlystopping"])
            if "trainer.checkpoint" in config:
                self.checkpoint.update_from_config(config["trainer.checkpoint"])
            if "trainer.optimizer" in config:
                self.optimizer.update_from_config(config["trainer.optimizer"])
            if "trainer.criteria" in config:
                self.criteria.update_from_config(config["trainer.criteria"])
            if "trainer.lr_scheduler" in config:
                self.lr_scheduler.update_from_config(config["trainer.lr_scheduler"])

        def __str__(self):
            return "{{split_data: {}, use_data: {}, use_model: {}, batch_size: {}, max_iter: {}, earlystopping: {}, checkpoint: {}, optimizer: {}, criteria: {}, lr_scheduler: {}}}".format(
                self.split_data,
                self.use_data,
                self.use_model,
                self.batch_size,
                self.max_iter,
                self.early_stopping,
                self.checkpoint,
                self.optimizer,
                self.criteria,
                self.lr_scheduler,
            )

        class EarlyStopping(object):
            def __init__(self):
                self.patience = 200
                self.min_delta = 0

            def update_from_config(self, config):
                if "patience" in config:
                    self.patience = str_to_number(config["patience"])
                if "min_delta" in config:
                    self.min_delta = str_to_number(config["min_delta"])

            def __str__(self):
                return "{{patience: {}, min_delta: {}}}".format(
                    self.patience, self.min_delta
                )

        class CheckPointing(object):
            def __init__(self):
                self.dir_path = "uq_testdata_ckpt/en_ckpt"
                self.warmup_epochs = 0

            def update_from_config(self, config):
                if "dir_path" in config:
                    self.dir_path = config["dir_path"]
                if "warmup_epochs" in config:
                    self.warmup_epochs = int(config["warmup_epochs"])

            def __str__(self):
                return "{{dir_path: {}, warmup_epochs: {}}}".format(
                    self.dir_path, self.warmup_epochs
                )

        class Optimizer(object):
            def __init__(self):
                self.init_lr = 1e-03

            def update_from_config(self, config):
                if "init_lr" in config:
                    self.init_lr = str_to_number(config["init_lr"])

            def __str__(self):
                return "{{init_lr: {}}}".format(self.init_lr)

        class Criteria(object):
            def __init__(self):
                self.focal_gamma = 2

            def update_from_config(self, config):
                if "focal_gamma" in config:
                    self.focal_gamma = str_to_number(config["focal_gamma"])

            def __str__(self):
                return "{{focal_gamma: {}}}".format(self.focal_gamma)

        class LearningRateScheduler(object):
            def __init__(self):
                self.T_0 = 200
                self.T_mult = 2

            def update_from_config(self, config):
                if "T_0" in config:
                    self.T_0 = str_to_number(config["T_0"])
                if "T_mult" in config:
                    self.T_mult = str_to_number(config["T_mult"])

            def __str__(self):
                return "{{T_0: {}, T_mult: {}}}".format(self.T_0, self.T_mult)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-c config_file] [...]", description="Run an UQ experiment."
    )

    parser.add_argument("-c", "--config")

    return parser


def get_extended_argparser() -> argparse.ArgumentParser:
    parser = init_argparse()
    parser.add_argument(
        "-s", "--select", help="model selection criteria, best_f1 or median_f1"
    )
    return parser


def get_experiment_config(parser: argparse.ArgumentParser = None) -> ExperimentConfig:
    if not parser:
        parser = init_argparse()
    args = parser.parse_args()
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"config file {args.config} inaccessible")
        config = ExperimentConfig(args.config)
    else:
        config = ExperimentConfig()
    logger.info(f"Experiment config: {config}")
    return config


def get_single_model_selection_criteria(parser: argparse.ArgumentParser) -> str:
    selection_criteria = "best_f1"
    args = parser.parse_args()
    if args.select:
        selection_criteria = args.select
    else:
        selection_criteria = "best_f1"
    if selection_criteria not in ["best_f1", "median_f1", "random"]:
        raise ValueError("unsupported model selection criteria " + selection_criteria)
    return selection_criteria
