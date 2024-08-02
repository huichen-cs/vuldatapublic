import json
import logging
import numpy as np
import torch
import torchmetrics
from packaging import version
from .experiment import ExperimentConfig
from .stochastic_dropout_bert import StochasticDropoutBertClassifier
from .stochastic_metrics import softmax_batch, entropy_batch, softmax_all
from .sampling_metrics import (
    compute_sampling_entropy,
    compute_sampling_mutual_information,
)
from .uq_metrics import (
    brier_score_from_tensors,
    compute_binary_acc_vs_conf_from_tensors,
    compute_binary_metrics_vs_conf_from_tensors,
)


logger = logging.getLogger(__name__)


class DropoutDisentangledUq(object):
    def __init__(
        self,
        model: StochasticDropoutBertClassifier,
        dataloader: torch.utils.data.DataLoader,
        n_stochastic_passes: int,
        n_aleatoric_samples: int,
        device: torch.DeviceObjType = None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.n_stochastic_passes = n_stochastic_passes
        self.n_aleatoric_samples = n_aleatoric_samples
        if device:
            self.device = device
        else:
            if next(self.model.parameters()).get_device() >= 0:
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

    def compute_uq_logits(self):
        """
        Compute aleatoic, epistermic uncertainty (all in logits).
        """
        n_batches = len(self.dataloader)
        mu_batch_list, sigma_batch_list = [None] * n_batches, [None] * n_batches
        for batch_idx, batch in enumerate(self.dataloader):
            input_ids, attention_mask, targets = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)
            mu_sample_list, sigma_sample_list = [None] * self.n_stochastic_passes, [
                None
            ] * self.n_stochastic_passes
            self.model.eval()
            for idx in range(self.n_stochastic_passes):
                mu_tmp, sigma_tmp = self.model(input_ids, attention_mask)
                mu_sample_list[idx] = mu_tmp
                sigma_sample_list[idx] = sigma_tmp
            self.model.train()
            mu_batch_models = torch.stack(mu_sample_list, dim=2)
            sigma_batch_models = torch.stack(sigma_sample_list, dim=2)
            mu_batch_list[batch_idx] = mu_batch_models
            sigma_batch_list[batch_idx] = sigma_batch_models
        mu_batches = torch.cat(mu_batch_list, dim=0)
        sigma_batches = torch.cat(sigma_batch_list, dim=0)
        sigma_aleatoric = torch.sqrt(torch.mean(torch.square(sigma_batches), dim=-1))
        sigma_epistermic = torch.sqrt(torch.var(mu_batches, dim=-1))
        mu_mean = torch.mean(mu_batches, dim=-1)
        return mu_mean, sigma_aleatoric, sigma_epistermic, mu_batches, sigma_batches

    def compute_uq_from_logits(
        self,
        mu_mean: torch.Tensor,
        sigma_aleatoric: torch.Tensor,
        sigma_epistermic: torch.Tensor,
        mu_all: torch.Tensor,
        sigma_all: torch.Tensor,
        n_aleatoric_samples: int,
        return_mean_std=True,
    ):
        proba_std_aleatoric, proba_mean_aleatoric = softmax_batch(
            mu_mean,
            sigma_aleatoric,
            n_aleatoric_samples,
            passed_log_sigma=False,
            return_mean_std=return_mean_std,
        )
        proba_std_epistermic, proba_mean_epistermic = softmax_batch(
            mu_mean,
            sigma_epistermic,
            n_aleatoric_samples,
            passed_log_sigma=False,
            return_mean_std=return_mean_std,
        )
        proba_std, proba_all = softmax_all(
            mu_all,
            sigma_all,
            n_aleatoric_samples,
            passed_log_sigma=False,
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


class StochasticDropoutBertClassifierEvalautor(object):
    """Evaluate Bert classifier with Monte Carlo Dropout."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: StochasticDropoutBertClassifier,
        dataset: torch.utils.data.TensorDataset,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, config.trainer.batch_size
        )

    def _predict(self, dataloader, device):
        proba_pred_list, conf_pred_list, labels_pred_list, targets_list = [], [], [], []
        for batch in dataloader:
            input_ids, attention_masks, targets = batch
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            targets = targets.to(device)
            proba_pred, confidence_pred, labels_pred = self.model.predict(
                input_ids, attention_masks
            )
            proba_pred_list.append(proba_pred)
            conf_pred_list.append(confidence_pred)
            labels_pred_list.append(labels_pred)
            targets_list.append(targets)
        confs_pred_all = torch.cat(conf_pred_list, dim=0)
        labels_pred_all = torch.cat(labels_pred_list, dim=0)
        proba_pred_all = torch.cat(proba_pred_list, dim=0)
        targets_all = torch.cat(targets_list)
        return proba_pred_all, confs_pred_all, labels_pred_all, targets_all

    def set_zero_to_nextafter(self, proba: torch.Tensor):
        proba[proba == 0] = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        return proba

    def get_one_hot_label(self, labels, num_classes=None):
        if num_classes:
            return torch.nn.functional.one_hot(labels, num_classes)
        else:
            return torch.nn.functional.one_hot(labels)

    def curve_triplet_to_dict(self, xyt, keys):
        curve = dict()
        for v, k in zip(xyt, keys):
            v = v.cpu().numpy().tolist() if v.get_device() >= 0 else v.numpy().tolist()
            curve[k] = v
        return curve

    def compute_ece(self, predicted_proba_tensor, test_label_tensor):
        if version.parse(torchmetrics.__version__) > version.parse("0.11.4"):
            ece = torchmetrics.functional.classification.binary_calibration_error(
                predicted_proba_tensor[:, 1].contiguous(),
                test_label_tensor,
                n_bins=15,
                norm="l1",
            )
        else:
            updated_predicted_proba_tensor = self.set_zero_to_nextafter(
                predicted_proba_tensor[:, 1].contiguous()
            )
            ece = torchmetrics.functional.classification.binary_calibration_error(
                updated_predicted_proba_tensor, test_label_tensor, n_bins=15, norm="l1"
            )
        return ece

    def result_dict_to_json(self, result_dict):
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

    def _compute_uq_eval_metrics(
        self, config, proba_pred, labels_pred, targets, metrics_list=None
    ):
        proba_pred = proba_pred.to(config.device)
        labels_pred = labels_pred.to(config.device)
        targets = targets.to(config.device)

        acc = (
            torchmetrics.functional.classification.binary_accuracy(labels_pred, targets)
            .cpu()
            .item()
        )
        precision = (
            torchmetrics.functional.classification.binary_precision(
                labels_pred, targets
            )
            .cpu()
            .item()
        )
        recall = (
            torchmetrics.functional.classification.binary_recall(labels_pred, targets)
            .cpu()
            .item()
        )
        f1 = (
            torchmetrics.functional.classification.binary_f1_score(labels_pred, targets)
            .cpu()
            .item()
        )
        mcc = (
            torchmetrics.functional.classification.binary_matthews_corrcoef(
                labels_pred, targets
            )
            .cpu()
            .item()
        )
        cmtx = torchmetrics.functional.classification.binary_confusion_matrix(
            labels_pred, targets
        )

        proba_contiguous = proba_pred[:, 1].contiguous()
        (
            prc_precision,
            prc_recall,
            prc_thresholds,
        ) = torchmetrics.functional.classification.binary_precision_recall_curve(
            proba_contiguous, targets, thresholds=20
        )
        prc = self.curve_triplet_to_dict(
            (prc_precision, prc_recall, prc_thresholds),
            ("precision", "recall", "thresholds"),
        )

        auprc = (
            torchmetrics.functional.classification.binary_average_precision(
                proba_contiguous, targets, thresholds=None
            )
            .cpu()
            .item()
        )
        (
            roc_fpr,
            roc_tpr,
            roc_thresholds,
        ) = torchmetrics.functional.classification.binary_roc(
            proba_contiguous, targets, thresholds=20
        )
        roc = self.curve_triplet_to_dict(
            (roc_fpr, roc_tpr, roc_thresholds), ("fpr", "tpr", "thresholds")
        )

        auroc = (
            torchmetrics.functional.classification.binary_auroc(
                proba_contiguous, targets, thresholds=None
            )
            .cpu()
            .item()
        )

        ece = self.compute_ece(proba_pred, targets)
        score = brier_score_from_tensors(
            self.get_one_hot_label(targets, num_classes=2), proba_pred
        )
        conf_thresholds = torch.linspace(0, 10, 11) * 0.1
        if metrics_list:
            metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
                proba_pred,
                targets,
                thresholds=conf_thresholds,
                metrics_list=metrics_list,
            )
        else:
            acc_list, count_list = compute_binary_acc_vs_conf_from_tensors(
                proba_pred, targets, thresholds=conf_thresholds
            )

        cmtx = (
            cmtx.cpu().numpy().tolist()
            if cmtx.get_device() >= 0
            else cmtx.numpy().tolist()
        )

        result_dict = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "cmtx": cmtx,
            "brier score": score,
            "auroc": auroc,
            "roc": roc,
            "auprc": auprc,
            "prc": prc,
            "ece": ece,
            "conf_thresholds": conf_thresholds,
            "im_ratio": config.data.imbalance_ratio,
            "sigma": config.datashift.sigma,
            "targets": targets.cpu().tolist(),
            "predictions": labels_pred.cpu().tolist(),
        }

        if metrics_list:
            for metric_name in metrics_list:
                result_dict[metric_name + "_list"] = [
                    a.item() for a in metrics_dict[metric_name]
                ]
        else:
            result_dict["acc_list"] = [a.item() for a in acc_list]
        result_dict["count_list"] = count_list
        logger.info("result_dict = {}".format(self.result_dict_to_json(result_dict)))
        return result_dict

    def compute_eval_metrics(self, config: ExperimentConfig):
        self.model = self.model.to(config.device)
        probas_pred, confs_pred, labels_pred, targets = self._predict(
            self.dataloader, self.config.device
        )
        result_dict = self._compute_uq_eval_metrics(
            config,
            probas_pred,
            labels_pred,
            targets,
            metrics_list=["acc", "precision", "recall", "f1", "mcc", "auprc", "auroc"],
        )

        uq_list = []
        if hasattr(config, "n_stochastic_passes"):
            n_stochastic_passes = config.n_stochastic_passes
        else:
            n_stochastic_passes = 100
        uq = DropoutDisentangledUq(
            self.model,
            self.dataloader,
            n_stochastic_passes,
            config.trainer.aleatoric_samples,
            device=config.device,
        )
        (
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
        ) = uq.compute_uq()
        for idx in range(len(targets)):
            target = targets[idx].item()
            label_pred = labels_pred[idx].item()
            label_conf = confs_pred[idx].item()
            proba_ale = proba_mean_aleatoric[idx].detach().cpu().numpy().tolist()
            proba_ale_std = proba_std_aleatoric[idx].item()
            proba_epi = proba_mean_epistermic[idx].detach().cpu().numpy().tolist()
            proba_epi_std = proba_std_epistermic[idx].item()
            entropy_ale = entropy_aleatoric[idx].item()
            entropy_epi = entropy_epistermic[idx].item()
            proba_std_instance = proba_std[idx].item()
            entropy_instance = entropy_all[idx].item()
            muinfo_instance = muinfo_all[idx].item()
            logits_mu_mean = mu_mean[idx].detach().cpu().numpy().tolist()
            logits_sigma_aleatoric = (
                sigma_aleatoric[idx].detach().cpu().numpy().tolist()
            )
            logits_sigma_epistermic = (
                sigma_epistermic[idx].detach().cpu().numpy().tolist()
            )
            if target == label_pred and target == 1:
                quadrant = "TP"
            elif target == label_pred and target == 0:
                quadrant = "TN"
            elif target != label_pred and label_pred == 1:
                quadrant = "FP"
            else:
                quadrant = "FN"
            uq_dict = {
                "index": idx,
                "target": target,
                "label_pred": label_pred,
                "label_conf": label_conf,
                "proba_ale": proba_ale,
                "proba_ale_std": proba_ale_std,
                "entropy_ale": entropy_ale,
                "proba_epi": proba_epi,
                "proba_epi_std": proba_epi_std,
                "entropy_epi": entropy_epi,
                "proba_std": proba_std_instance,
                "entropy": entropy_instance,
                "muinfo": muinfo_instance,
                "mu_mean": logits_mu_mean,
                "sigma_aleatoric": logits_sigma_aleatoric,
                "sigma_epistermic": logits_sigma_epistermic,
                "quadrant": quadrant,
            }
            uq_list.append(uq_dict)
        result_dict["uq"] = uq_list
        return result_dict
