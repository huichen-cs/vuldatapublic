import json
import gc
import logging
import torch.multiprocessing as mp
import numpy as np
import os
import torch
import torchmetrics
from packaging import version

# from typing import Tuple
# from .calibration_error import binary_calibration_error
from uqmodel.stochasticbert.logging_utils import init_logging, get_global_logfilename
from uqmodel.stochasticbert.ensemble_mlc import StochasticEnsembleClassifier
from uqmodel.stochasticbert.ensemble_trainer import EnsembleTrainer
from uqmodel.stochasticbert.checkpoint import EnsembleCheckpoint
from uqmodel.stochasticbert.sampling_metrics import (
    compute_sampling_entropy,
    compute_sampling_mutual_information,
)
from uqmodel.stochasticbert.uq_metrics import (
    compute_binary_acc_vs_conf_from_tensors,
    compute_binary_metrics_vs_conf_from_tensors,
    brier_score_from_tensors,
)
from uqmodel.stochasticbert.stochastic_metrics import (
    softmax_batch,
    entropy_batch,
    softmax_all,
)
from uqmodel.stochasticbert.ensemble_mlc import EnsembleClassifier
from uqmodel.stochasticbert.dropout_mlc import DropoutClassifier
from uqmodel.stochasticbert.vanilla_mlc import VanillaClassifier
from uqmodel.stochasticbert.stochastic_mlc import StochasticMultiLayerClassifier
from uqmodel.stochasticbert.experiment_new import ExperimentConfig
from uqmodel.stochasticbert.datashift import (
    DataShift,
    ShiftedFeatureDataSet,
    PortionShiftedFeatureDataSet,
)


logger = logging.getLogger(__name__)


def get_one_hot_label(labels, num_classes=None, device=None):
    if num_classes:
        return torch.nn.functional.one_hot(labels, num_classes).to(device)
    else:
        return torch.nn.functional.one_hot(labels).to(device)


def load_from_checkpoint_with_datashift(
    model_type: str, config: ExperimentConfig, device: torch.DeviceObjType = None
):
    num_workers = (
        os.cpu_count()
        if os.cpu_count() < config.trainer.max_dataloader_workers
        else config.trainer.max_dataloader_workers
    )
    ckpt = EnsembleCheckpoint(
        config.trainer.checkpoint.dir_path,
        warmup_epochs=config.trainer.checkpoint.warmup_epochs,
    )

    try:
        train_dataset, val_dataset, test_dataset, ps_columns = ckpt.load_datasets()
        logger.info(
            "loaded train/val/test datasets from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        )
    except FileNotFoundError as err:
        logger.error(
            "failed to load train/va/test datasets from checkpoint at {}".format(
                config.trainer.checkpoint.dir_path
            )
        )
        raise err

    if not config.datashift:
        raise ValueError("configuration does not have data shift setup")

    datashift = DataShift.from_dict(config.datashift.param_dict)
    if config.data.shift_data_portion:
        (train_dataset, val_dataset, test_dataset) = (
            PortionShiftedFeatureDataSet(
                train_dataset, datashift, config.data.shift_data_portion
            ),
            PortionShiftedFeatureDataSet(
                val_dataset, datashift, config.data.shift_data_portion
            ),
            PortionShiftedFeatureDataSet(
                test_dataset, datashift, config.data.shift_data_portion
            ),
        )
        logger.info(
            "portion {} data are applied data shiftat sigma".format(
                config.datashift.sigma
            )
        )
    else:
        (train_dataset, val_dataset, test_dataset) = (
            ShiftedFeatureDataSet(train_dataset, datashift),
            ShiftedFeatureDataSet(val_dataset, datashift),
            ShiftedFeatureDataSet(test_dataset, datashift),
        )
        logger.info(
            "all data are applied data shift at sigma {}".format(config.datashift.sigma)
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    if model_type == "disentangle":
        ouput_log_sigma = False
        ensemble = StochasticEnsembleClassifier(
            StochasticMultiLayerClassifier,
            ouput_log_sigma,  # log_sigma <- False
            config.model.ensemble_size,
            len(ps_columns),
            2,
            neurons=config.model.num_neurons,
            dropouts=config.model.dropout_ratios,
            activation=torch.nn.LeakyReLU(),
        )
        trainer = EnsembleTrainer(
            ensemble,
            criteria=None,
            lr_scheduler=None,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=None,
            n_samples=config.trainer.aleatoric_samples,
            ouput_log_sigma=ouput_log_sigma,
        )
    elif model_type == "predictive":
        ensemble = EnsembleClassifier(
            config.model.ensemble_size,
            len(ps_columns),
            2,
            neurons=config.model.num_neurons,
            dropouts=config.model.dropout_ratios,
            activation=torch.nn.LeakyReLU(),
        )
        trainer = EnsembleTrainer(
            ensemble,
            criteria=None,
            lr_scheduler=None,
            max_iter=config.trainer.max_iter,
            init_lr=config.trainer.optimizer.init_lr,
            device=device,
            checkpoint=ckpt,
            earlystopping=None,
        )
    else:
        raise ValueError(
            "model type {} is not in [disentangle, predictive]".format(model_type)
        )

    try:
        ensemble = trainer.load_checkpoint()
        logger.info("use the ensemble model from checkpoint, no training")
    except FileNotFoundError as err:
        raise ValueError(
            "failed to load pre-trained ensemble model from checkpoint at {}".format(
                config.trainer_checkpoint_dir_path
            )
        ) from err
    return train_dataloader, val_dataloader, test_dataloader, ensemble


def set_zero_to_nextafter(proba: torch.Tensor):
    proba[proba == 0] = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    return proba


def compute_ece(predicted_proba_tensor, test_label_tensor):
    if version.parse(torchmetrics.__version__) > version.parse("0.11.4"):
        ece = torchmetrics.functional.classification.binary_calibration_error(
            predicted_proba_tensor[:, 1].contiguous(),
            test_label_tensor,
            n_bins=15,
            norm="l1",
        )
    else:
        updated_predicted_proba_tensor = set_zero_to_nextafter(
            predicted_proba_tensor[:, 1].contiguous()
        )
        # ece = binary_calibration_error(updated_predicted_proba_tensor, test_label_tensor, n_bins=15, norm='l1')
        ece = torchmetrics.functional.classification.binary_calibration_error(
            updated_predicted_proba_tensor, test_label_tensor, n_bins=15, norm="l1"
        )
    return ece


def curve_tensors_to_list(xyt, keys):
    curve = dict()
    for v, k in zip(xyt, keys):
        v = v.cpu().numpy().tolist() if v.get_device() >= 0 else v.numpy().tolist()
        curve[k] = v
    return curve


def compute_uq_eval_metrics(
    config,
    predicted_proba_tensor,
    predicted_label_tensor,
    test_label_tensor,
    py_script=None,
    metrics_list=None,
):
    acc = torchmetrics.functional.classification.binary_accuracy(
        predicted_label_tensor, test_label_tensor
    )
    precision = torchmetrics.functional.classification.binary_precision(
        predicted_label_tensor, test_label_tensor
    )
    recall = torchmetrics.functional.classification.binary_recall(
        predicted_label_tensor, test_label_tensor
    )
    f1 = torchmetrics.functional.classification.binary_f1_score(
        predicted_label_tensor, test_label_tensor
    )
    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(
        predicted_label_tensor, test_label_tensor
    )
    cmtx = torchmetrics.functional.classification.binary_confusion_matrix(
        predicted_label_tensor, test_label_tensor
    )
    proba_contiguous = predicted_proba_tensor[:, 1].contiguous()
    (
        prc_precision,
        prc_recall,
        prc_thresholds,
    ) = torchmetrics.functional.classification.binary_precision_recall_curve(
        proba_contiguous, test_label_tensor, thresholds=20
    )
    auprc = torchmetrics.functional.classification.binary_average_precision(
        proba_contiguous, test_label_tensor, thresholds=None
    )
    (
        roc_fpr,
        roc_tpr,
        roc_thresholds,
    ) = torchmetrics.functional.classification.binary_roc(
        proba_contiguous, test_label_tensor, thresholds=20
    )
    auroc = torchmetrics.functional.classification.binary_auroc(
        proba_contiguous, test_label_tensor, thresholds=None
    )
    # ece = torchmetrics.functional.classification.binary_calibration_error(predicted_confidence_tensor, test_label_tensor, n_bins=15, norm='l1')
    # ece = torchmetrics.functional.classification.binary_calibration_error(predicted_proba_tensor[:, 1].contiguous(), test_label_tensor, n_bins=15, norm='l1')
    # ece = binary_calibration_error(predicted_proba_tensor[:, 1].contiguous(), test_label_tensor, n_bins=15, norm='l1')
    # updated_predicted_proba_tensor = set_zero_to_nextafter(predicted_proba_tensor[:, 1].contiguous())
    # # ece = binary_calibration_error(updated_predicted_proba_tensor, test_label_tensor, n_bins=15, norm='l1')
    # ece = torchmetrics.functional.classification.binary_calibration_error(updated_predicted_proba_tensor, test_label_tensor, n_bins=15, norm='l1')
    ece = compute_ece(predicted_proba_tensor, test_label_tensor)
    score = brier_score_from_tensors(
        get_one_hot_label(test_label_tensor, num_classes=2), predicted_proba_tensor
    )
    conf_thresholds = torch.linspace(0, 10, 11) * 0.1
    if metrics_list:
        metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
            predicted_proba_tensor,
            test_label_tensor,
            thresholds=conf_thresholds,
            metrics_list=metrics_list,
        )
    else:
        acc_list, count_list = compute_binary_acc_vs_conf_from_tensors(
            predicted_proba_tensor, test_label_tensor, thresholds=conf_thresholds
        )
    logger.debug("unique test labels: {}".format(test_label_tensor.unique()))
    logger.debug("y -> {}".format(test_label_tensor))
    logger.debug("y_pred -> {}".format(predicted_label_tensor))
    logger.debug("y | y == 1 -> {}".format(test_label_tensor[test_label_tensor == 1]))
    logger.debug(
        "y_pred | y == 1 -> {}".format(predicted_label_tensor[test_label_tensor == 1])
    )
    logger.debug(f"acc: {acc}")
    logger.debug(f"precision:{precision}")
    logger.debug(f"recall:{recall}")
    logger.debug(f"f1:{f1}")
    logger.debug(f"mcc:{mcc}")
    logger.debug(f"prc: ({prc_precision}, {prc_recall}, {prc_thresholds})")
    logger.debug(f"auprc: {auprc}")
    logger.debug(f"roc: ({roc_fpr, roc_tpr, roc_thresholds})")
    logger.debug(f"auroc: {auroc}")
    logger.debug(f"cmtx:{cmtx}")
    logger.debug(f"ece:{ece}")
    logger.debug(f"brier score: {score}")

    cmtx = (
        cmtx.cpu().numpy().tolist() if cmtx.get_device() >= 0 else cmtx.numpy().tolist()
    )
    prc = curve_tensors_to_list(
        (prc_precision, prc_recall, prc_thresholds),
        ("precision", "recall", "thresholds"),
    )
    roc = curve_tensors_to_list(
        (roc_fpr, roc_tpr, roc_thresholds), ("fpr", "tpr", "thresholds")
    )
    result_dict = {
        "acc": acc.cpu(),
        "precision": precision.cpu(),
        "recall": recall.cpu(),
        "f1": f1.cpu(),
        "mcc": mcc.cpu(),
        "cmtx": cmtx,
        "brier score": score.cpu(),
        "auroc": auroc.cpu(),
        "roc": roc,
        "auprc": auprc.cpu(),
        "prc": prc,
        "ece": ece.cpu(),
        "conf_thresholds": conf_thresholds.cpu(),
        "im_ratio": config.imbalance_ratio,
        "sigma": 0,
        "py": py_script,
        "config_fn": config.config_fn,
        "cmdline": "",
        "targets": test_label_tensor.cpu().tolist(),
        "predictions": predicted_label_tensor.cpu().tolist(),
    }

    if metrics_list:
        for metric_name in metrics_list:
            result_dict[metric_name + "_list"] = [
                a.item() for a in metrics_dict[metric_name]
            ]
    else:
        result_dict["acc_list"] = [a.item() for a in acc_list]
    result_dict["count_list"] = count_list
    logger.info("result_dict = {}".format(result_dict_to_json(result_dict)))

    return result_dict


def result_dict_to_json(result_dict):
    for k in result_dict.keys():
        result_dict[k] = (
            result_dict[k].cpu().numpy()
            if isinstance(result_dict[k], torch.Tensor)
            else result_dict[k]
        )
        result_dict[k] = (
            result_dict[k].tolist()
            if isinstance(result_dict[k], np.ndarray)
            else result_dict[k]
        )
    return json.dumps(result_dict, indent=2)


class EnsembleUq(object):
    def __init__(
        self, ensemble: EnsembleClassifier, data_loader: torch.utils.data.DataLoader
    ):
        self.ensemble = ensemble
        self.ensemble_size = len(ensemble)
        self.data_loader = data_loader

    def compute_uq(self):
        logits, proba = self.ensemble.predict_logits_proba(self.data_loader)
        entropy = compute_sampling_entropy(proba)
        mutual_info = compute_sampling_mutual_information(proba)
        return logits, proba, entropy, mutual_info


class DropoutUq(object):
    def __init__(
        self,
        dropout_model: DropoutClassifier,
        data_loader: torch.utils.data.DataLoader,
        n_stochastic_passes: int = 1000,
        keep_samples: bool = False,
    ):
        self.dropout_model = dropout_model
        self.n_stochastic_passes = n_stochastic_passes
        self.data_loader = data_loader
        self.keep_samples = keep_samples

    def compute_uq(self):
        logits, proba = self.dropout_model.predict_logits_proba(
            self.data_loader, self.n_stochastic_passes
        )
        entropy = compute_sampling_entropy(proba)
        mutual_info = compute_sampling_mutual_information(proba)
        return logits, proba, entropy, mutual_info


class VanillaUq(object):
    def __init__(
        self, vanilla_model: VanillaClassifier, data_loader: torch.utils.data.DataLoader
    ):
        self.vanilla_model = vanilla_model
        self.data_loader = data_loader

    def compute_uq(self):
        logits, proba = self.vanilla_model.predict_logits_proba(self.data_loader)
        logits = logits.unsqueeze(-2)
        proba = proba.unsqueeze(-2)
        entropy = compute_sampling_entropy(proba)
        mutual_info = compute_sampling_mutual_information(proba)
        return logits, proba, entropy, mutual_info


class EnsembleDisentangledUq(object):
    def __init__(
        self,
        ensemble: StochasticEnsembleClassifier,
        data_loader: torch.utils.data.DataLoader,
        n_aleatoric_samples: int,
        device: torch.device = None,
        mp: bool = True,
    ):
        self.ensemble = ensemble
        self.ensemble_size = len(ensemble)
        self.data_loader = data_loader
        self.n_aleatoric_samples = n_aleatoric_samples
        self.mp = mp
        if device:
            self.device = device
        else:
            if next(self.ensemble[0].parameters()).get_device() >= 0:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    # def compute_uq_logits(self):
    #     """
    #     compute aleatoic, epistermic uncertainty (all in logits)
    #     """
    #     logits = self.ensemble.predict(self.dataloader)
    #     return logits

    def _fit(self, model_idx, batch_idx, model, input_ids, attention_mask, device=None):
        torch.cuda.empty_cache()
        if device is None:
            logger.debug("eval for model {} with batch {}".format(model_idx, batch_idx))
            model = model.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            model.eval()
            mu, sigma = model(input_ids=input_ids, attention_mask=attention_mask)
            model.train()
            return mu, sigma
        else:
            mu, sigma = self._cpu_batch_fit(
                model_idx, batch_idx, model, input_ids, attention_mask
            )
            return mu, sigma

    def _batch_fit(
        self,
        model_idx,
        batch_idx,
        model,
        input_ids,
        attention_mask,
        logfilename,
        rtn_dict,
    ):
        init_logging(logfilename, append=True)
        logger.debug("eval for model {} with batch {}".format(model_idx, batch_idx))
        # print('eval for model {} with batch {}'.format(model_idx, batch_idx))
        model = model.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        model.eval()
        mu_tmp, sigma_tmp = model(input_ids=input_ids, attention_mask=attention_mask)
        rtn_dict["mu"] = mu_tmp.detach().cpu()
        rtn_dict["sigma"] = sigma_tmp.detach().cpu()
        model.train()

    def _cpu_batch_fit(self, model_idx, batch_idx, model, input_ids, attention_mask):
        logger.debug("eval for model {} with batch {}".format(model_idx, batch_idx))
        # device = torch.device("cpu")
        # model = model.to(device)
        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        model.eval()
        mu_tmp, sigma_tmp = model(input_ids=input_ids, attention_mask=attention_mask)
        model.train()
        gc.collect()
        return mu_tmp, sigma_tmp

    def _mp_fit(
        self, model_idx, batch_idx, model, input_ids, attention_mask, device=None
    ):
        torch.cuda.empty_cache()
        if device is None:
            mp.set_sharing_strategy("file_system")
            # mp.set_sharing_strategy('file_descriptor')
            mp.set_start_method(method="forkserver", force=True)
            manager = mp.Manager()
            rtn_dict = manager.dict()
            # ctx = mp.get_context('spawn')
            ctx = mp.get_context("forkserver")
            p = ctx.Process(
                target=self._batch_fit,
                args=(
                    model_idx,
                    batch_idx,
                    model,
                    input_ids,
                    attention_mask,
                    get_global_logfilename(),
                    rtn_dict,
                ),
            )
            p.start()
            p.join()
            mu, sigma = rtn_dict["mu"], rtn_dict["sigma"]
            return mu, sigma
        else:
            mu, sigma = self._cpu_batch_fit(
                model_idx, batch_idx, model, input_ids, attention_mask
            )
            return mu, sigma

    def _slow_compute_uq_logits(self):
        """
        Compute aleatoic, epistermic uncertainty (all in logits).
        """
        n_batches = len(self.data_loader)
        mu_batch_list, sigma_batch_list = [None] * n_batches, [None] * n_batches
        # mu_batch_list, sigma_batch_list = [None]*6, [None]*6
        for batch_idx, batch in enumerate(self.data_loader):
            # if batch_idx >=6:
            #     break
            input_ids, attention_mask, _ = batch
            # input_ids = input_ids.to(self.device)
            # attention_mask = attention_mask.to(self.device)
            # targets = targets.to(self.device)

            mu_model_list, sigma_model_list = [None] * self.ensemble_size, [
                None
            ] * self.ensemble_size
            # mu_model_list, sigma_model_list = [None]*3, [None]*3
            for model_idx, model in enumerate(self.ensemble):
                # if model_idx > 2:
                #     break
                # model = model.to(self.device)
                # model.eval()
                # mu_tmp, sigma_tmp = model(input_ids=input_ids, attention_mask=attention_mask)
                # mu_model_list[model_idx] = mu_tmp
                # sigma_model_list[model_idx] = sigma_tmp
                # model.train()
                (mu_model_list[model_idx], sigma_model_list[model_idx]) = self._mp_fit(
                    model_idx, batch_idx, model, input_ids, attention_mask
                )
            mu_batch_models = torch.stack(mu_model_list, dim=2)
            sigma_batch_models = torch.stack(sigma_model_list, dim=2)
            mu_batch_list[batch_idx] = mu_batch_models
            sigma_batch_list[batch_idx] = sigma_batch_models
        mu_batches = torch.cat(mu_batch_list, dim=0)
        if self.ensemble.log_sigma:
            log_sigma_batches = torch.cat(sigma_batch_list, dim=0)
            sigma_batches = torch.exp(log_sigma_batches).mean(dim=-1)
        else:
            sigma_batches = torch.cat(sigma_batch_list, dim=0)
        sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=-1))
        sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=-1))
        mu_mean = torch.mean(mu_batches, dim=-1)
        return mu_mean, sigma_aleatoric, sigma_epistermic, mu_batches, sigma_batches

    def _member_model_logits(self, dataloader, model, n_batches, device, rtn_dict):
        model = model.to(device)
        model.eval()
        mu_batch_list, sigma_batch_list = [None] * n_batches, [None] * n_batches
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            (mu_batch_list[batch_idx], sigma_batch_list[batch_idx]) = model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            input_ids, attention_mask = None, None
        mu_batches = torch.cat(mu_batch_list, dim=0)
        sigma_batches = torch.cat(sigma_batch_list, dim=0)
        model.train()
        model = model.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        gc.collect()
        rtn_dict["mu_batches"], rtn_dict["sigma_batches"] = mu_batches, sigma_batches

    def _mp_member_model_logits(self, dataloader, model, n_batches, device):
        mp.set_sharing_strategy("file_system")
        manager = mp.Manager()
        rtn_dict = manager.dict()
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=self._member_model_logits,
            args=(dataloader, model, n_batches, device, rtn_dict),
        )
        p.start()
        p.join()
        mu_batches = rtn_dict["mu_batches"].detach().cpu()
        sigma_batches = rtn_dict["sigma_batches"].detach().cpu()
        return mu_batches, sigma_batches

    def _member_model_logits_q(self, dataloader, model, n_batches, device, queue):
        model = model.to(device)
        model.eval()
        # mu_batch_list, sigma_batch_list = [None]*n_batches, [None]*n_batches
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            mu, sigma = model(input_ids=input_ids, attention_mask=attention_mask)
            queue.put((batch_idx, mu, sigma))
            print(queue.empty())
            input_ids, attention_mask = None, None
        # mu_batches = torch.cat(mu_batch_list, dim=0)
        # sigma_batches = torch.cat(sigma_batch_list, dim=0)
        model.train()
        # queue.put((mu_batches, sigma_batches))

    def _mp_member_model_logits_q(self, dataloader, model, n_batches, device):
        mp.set_sharing_strategy("file_system")
        manager = mp.Manager()
        queue = manager.Queue()
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=self._member_model_logits_q,
            args=(dataloader, model, n_batches, device, queue),
        )
        p.start()
        p.join()
        n_batches = len(dataloader)
        mu_batch_list, sigma_batch_list = [None] * n_batches, [None] * n_batches
        print(queue.empty())
        while not queue.empty():
            batch_idx, mu, sigma = queue.get()
            mu_batch_list[batch_idx], sigma_batch_list[batch_idx] = (
                mu.detach().cpu(),
                sigma.detach().cpu(),
            )
        mu_batches = torch.cat(mu_batch_list, dim=0)
        sigma_batches = torch.cat(sigma_batch_list, dim=0)
        return mu_batches, sigma_batches

    def _compute_uq_logits_q(self):
        """
        Compute aleatoic, epistermic uncertainty (all in logits).
        """
        n_batches = len(self.data_loader)
        mu_model_list, sigma_model_list = [None] * self.ensemble_size, [
            None
        ] * self.ensemble_size
        for model_idx, model in enumerate(self.ensemble):
            mu_batches, sigma_batches = self._mp_member_model_logits_q(
                self.data_loader, model, n_batches, self.device
            )

            mu_model_list[model_idx] = mu_batches
            sigma_model_list[model_idx] = sigma_batches
        mu_batches = torch.stack(mu_model_list, dim=0)
        if self.ensemble.log_sigma:
            log_sigma_batches = torch.stack(sigma_model_list, dim=0)
            sigma_batches = torch.exp(log_sigma_batches).mean(dim=-1)
        else:
            sigma_batches = torch.stack(sigma_model_list, dim=0)
        sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=-1))
        sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=-1))
        mu_mean = torch.mean(mu_batches, dim=-1)
        return mu_mean, sigma_aleatoric, sigma_epistermic, mu_batches, sigma_batches

    def fast_compute_uq_logits(self):
        """
        Compute aleatoic, epistermic uncertainty (all in logits).
        """
        torch.cuda.empty_cache()
        gc.collect()
        n_batches = len(self.data_loader)
        mu_model_list, sigma_model_list = [None] * self.ensemble_size, [
            None
        ] * self.ensemble_size
        for model_idx, model in enumerate(self.ensemble):
            mu_batches, sigma_batches = self._mp_member_model_logits(
                self.data_loader, model, n_batches, self.device
            )

            mu_model_list[model_idx] = mu_batches
            sigma_model_list[model_idx] = sigma_batches
        mu_batches = torch.stack(mu_model_list, dim=0)
        if self.ensemble.log_sigma:
            log_sigma_batches = torch.stack(sigma_model_list, dim=0)
            sigma_batches = torch.exp(log_sigma_batches).mean(dim=-1)
        else:
            sigma_batches = torch.stack(sigma_model_list, dim=0)
        sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=-1))
        sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=-1))
        mu_mean = torch.mean(mu_batches, dim=-1)
        return mu_mean, sigma_aleatoric, sigma_epistermic, mu_batches, sigma_batches

    def compute_uq_logits(self):
        """
        Compute aleatoic, epistermic uncertainty (all in logits).
        """
        n_batches = len(self.data_loader)
        mu_model_list, sigma_model_list = [None] * self.ensemble_size, [
            None
        ] * self.ensemble_size
        # mu_model_list, sigma_model_list = [None]*3, [None]*3
        for model_idx, model in enumerate(self.ensemble):
            # if model_idx > 2:
            #     break
            mu_batch_list, sigma_batch_list = [None] * n_batches, [None] * n_batches
            # mu_batch_list, sigma_batch_list = [None]*6, [None]*6
            for batch_idx, batch in enumerate(self.data_loader):
                # if batch_idx >= 6:
                #     break
                input_ids, attention_mask, _ = batch
                if self.mp:
                    (
                        mu_batch_list[batch_idx],
                        sigma_batch_list[batch_idx],
                    ) = self._mp_fit(
                        model_idx, batch_idx, model, input_ids, attention_mask
                    )
                else:
                    (mu_batch_list[batch_idx], sigma_batch_list[batch_idx]) = self._fit(
                        model_idx, batch_idx, model, input_ids, attention_mask
                    )
            mu_batches = torch.cat(mu_batch_list, dim=0)
            sigma_batches = torch.cat(sigma_batch_list, dim=0)
            mu_model_list[model_idx] = mu_batches
            sigma_model_list[model_idx] = sigma_batches
        mu_batches = torch.stack(mu_model_list, dim=0)
        if self.ensemble.log_sigma:
            log_sigma_batches = torch.stack(sigma_model_list, dim=0)
            sigma_batches = torch.exp(log_sigma_batches).mean(dim=-1)
        else:
            sigma_batches = torch.stack(sigma_model_list, dim=0)
        # sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=-1))
        # sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=-1))
        # mu_mean = torch.mean(mu_batches, dim=-1)
        sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=0))
        sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=0))
        mu_mean = torch.mean(mu_batches, dim=0)
        mu_batches = torch.transpose(mu_batches, 0, -1).transpose(1, 0)
        sigma_batches = torch.transpose(sigma_batches, 0, -1).transpose(1, 0)
        return mu_mean, sigma_aleatoric, sigma_epistermic, mu_batches, sigma_batches

    def compute_uq_from_logits(
        self,
        mu_mean: torch.Tensor,
        sigma_aleatoric: torch.Tensor,
        sigma_epistermic: torch.Tensor,
        mu_all: torch.Tensor,
        sigma_all: torch.Tensor,
        n_samples: int,
        return_mean_std=True,
    ):
        proba_std_aleatoric, proba_mean_aleatoric = softmax_batch(
            mu_mean,
            sigma_aleatoric,
            n_samples,
            passed_log_sigma=self.ensemble.log_sigma,
            return_mean_std=return_mean_std,
        )
        proba_std_epistermic, proba_mean_epistermic = softmax_batch(
            mu_mean,
            sigma_epistermic,
            n_samples,
            passed_log_sigma=self.ensemble.log_sigma,
            return_mean_std=return_mean_std,
        )
        proba_std, proba_all = softmax_all(
            mu_all,
            sigma_all,
            n_samples,
            passed_log_sigma=self.ensemble.log_sigma,
            return_mean_std=return_mean_std,
        )
        return (
            proba_std_aleatoric,
            proba_mean_aleatoric,
            proba_std_epistermic,
            proba_mean_epistermic,
            proba_std,
            proba_all,
        )

    def compute_uq(self, return_mean_std=True):
        (
            mu_mean,
            sigma_aleatoric,
            sigma_epistermic,
            mu_all,
            sigma_all,
        ) = self.compute_uq_logits()
        (
            proba_std_aleatoric,
            proba_mean_aleatoric,
            proba_std_epistermic,
            proba_mean_epistermic,
            proba_std,
            proba_all,
        ) = self.compute_uq_from_logits(
            mu_mean,
            sigma_aleatoric,
            sigma_epistermic,
            mu_all,
            sigma_all,
            self.n_aleatoric_samples,
            return_mean_std=return_mean_std,
        )
        entropy_aleatoric = entropy_batch(proba_mean_aleatoric)
        entropy_epistermic = entropy_batch(proba_mean_epistermic)
        entropy_all = compute_sampling_entropy(proba_all)
        muinfo_all = compute_sampling_mutual_information(proba_all)
        return (
            proba_std_aleatoric,
            proba_mean_aleatoric,
            entropy_aleatoric,
            proba_std_epistermic,
            proba_mean_epistermic,
            entropy_epistermic,
            proba_std,
            entropy_all,
            muinfo_all,
            mu_mean,
            sigma_aleatoric,
            sigma_epistermic,
        )


def add_predictive_uq_to_result_dict(result_dict: dict, uq):
    logits_pred, proba_pred, entropy, mutual_info = uq.compute_uq()
    entropy = entropy.detach().cpu().numpy().tolist()
    mutual_info = mutual_info.detach().cpu().numpy().tolist()
    if isinstance(uq, DropoutUq) and not uq.keep_samples:
        uq_dict = {"entropy": entropy, "mutual_info": mutual_info}
    else:
        logits_pred = logits_pred.detach().cpu().numpy().tolist()
        proba_pred = proba_pred.detach().cpu().numpy().tolist()
        uq_dict = {
            "logits_pred": logits_pred,
            "proba_pred": proba_pred,
            "entropy": entropy,
            "mutual_info": mutual_info,
        }
    result_dict.update(uq_dict)
    return result_dict
