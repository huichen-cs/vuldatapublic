import packaging
import torch
import torchmetrics
from typing import Callable, Dict, List, Tuple, Union
from uqmodel.bert.experiment import ExperimentConfig
from uqmodel.bert.ensemble_bert import EnsembleBertClassifier
from uqmodel.bert.sampling_metrics import (
    compute_sampling_entropy,
    compute_sampling_mutual_information,
)


def set_zero_to_nextafter(proba: torch.Tensor) -> torch.Tensor:
    proba = proba.clone()
    proba[proba == 0] = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    return proba


def compute_ece(proba, targets):
    if packaging.version.parse(torchmetrics.__version__) > packaging.version.parse(
        "0.11.4"
    ):
        ece = torchmetrics.functional.classification.binary_calibration_error(
            proba, targets, n_bins=15, norm="l1"
        )
    else:
        proba = set_zero_to_nextafter(proba)
        ece = torchmetrics.functional.classification.binary_calibration_error(
            proba, targets, n_bins=15, norm="l1"
        )
    return ece


def brier_score(targets, probs):
    sum = torch.sum((targets - probs) ** 2)
    score = sum / len(targets)
    return score


def curve_triplet_to_dict(xyt, keys):
    curve = dict()
    for v, k in zip(xyt, keys):
        v = v.cpu().tolist() if v.get_device() >= 0 else v.tolist()
        curve[k] = v
    return curve


def get_one_hot_label(labels, num_classes=None):
    if num_classes:
        return torch.nn.functional.one_hot(labels, num_classes)
    else:
        return torch.nn.functional.one_hot(labels)


def compute_binary_metrics_vs_conf_from_tensors(
    test_proba_pred: torch.Tensor,
    test_label: torch.Tensor,
    thresholds: torch.Tensor = None,
    empty_is: float = 1.0,
    metrics_list: list = ("acc"),
) -> Tuple[Dict, List]:
    if thresholds is None:
        thresholds = torch.linspace(0, 10, 11) * 0.1
    metrics_func: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "acc": torchmetrics.functional.classification.binary_accuracy,
        "f1": torchmetrics.functional.classification.binary_f1_score,
        "recall": torchmetrics.functional.classification.binary_recall,
        "precision": torchmetrics.functional.classification.binary_precision,
        "mcc": torchmetrics.functional.classification.binary_matthews_corrcoef,
        "auprc": torchmetrics.functional.classification.binary_average_precision,
        "auroc": torchmetrics.functional.classification.binary_auroc,
    }
    if not metrics_list:
        metrics_list = ["acc"]
    metrics_dict: Dict = dict()
    for metric_name in metrics_list:
        if metric_name not in metrics_func.keys():
            raise ValueError("metrics must be in {}".format(metrics_func.keys()))
        metrics_dict[metric_name] = []
    count_list = []
    for t in thresholds:
        pred = test_proba_pred[test_proba_pred[:, 1] >= t]
        test = test_label[test_proba_pred[:, 1] >= t]
        count = len(test)
        for metric_name in metrics_list:
            if count > 0:
                metric_value = metrics_func[metric_name](pred[:, 1], test)
            else:
                metric_value = torch.tensor(empty_is)
            metrics_dict[metric_name].append(metric_value)
        count_list.append(count)
    return metrics_dict, count_list


class EnsembleBertClassifierEvalautor(object):
    """Evaluate Bert classifier."""

    def __init__(
        self,
        config: ExperimentConfig,
        ensemble: EnsembleBertClassifier,
        dataset: torch.utils.data.TensorDataset,
    ):
        self.ensemble = ensemble
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, config.trainer.batch_size
        )

    def predict_proba(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: Union[torch.device, str, None],
    ):
        ensemble_proba_list, mean_proba_list, confidence_list, labels_list = (
            [],
            [],
            [],
            [],
        )
        ensemble_model = self.ensemble.to(device)
        ensemble_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                ensemble_proba, mean_proba, confidence, labels = ensemble_model.predict(
                    input_ids, attention_mask
                )
                ensemble_proba_list.append(ensemble_proba)
                mean_proba_list.append(mean_proba)
                confidence_list.append(confidence)
                labels_list.append(labels)
        ensemble_model.train()
        return ensemble_proba_list, mean_proba_list, confidence_list, labels_list

    def get_targets(self, dataloader: torch.utils.data.DataLoader) -> List:
        targets = [t for _, _, t in dataloader]
        return targets

    def compute_eval_metrics(
        self, device: Union[torch.device, str, None] = None
    ) -> dict:
        """Compute evaluaton metrics.

        compute evaluaton metrics including
           predictive metrics: precision, recall, accuracy, F1, AUROC, AUPR, MCC
           UQ metrics: ECE, Brier score, entropy (total uncertainty), data uncertainty,
                       mutual information (epistemic)
        """
        dataloader = self.dataloader
        (
            ensemble_proba_list,
            mean_proba_list,
            confidence_list,
            labels_list,
        ) = self.predict_proba(dataloader, device)
        mean_proba = torch.cat(mean_proba_list)
        proba_pred = mean_proba[:, 1].to(device)
        labels_pred = torch.cat(labels_list).to(device)
        targets_list = self.get_targets(dataloader)
        targets = torch.cat(targets_list).to(device)
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
        acc = (
            torchmetrics.functional.classification.binary_accuracy(labels_pred, targets)
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
        cmtx = (
            torchmetrics.functional.classification.binary_confusion_matrix(
                labels_pred, targets
            )
            .cpu()
            .tolist()
        )
        (
            prc_precision,
            prc_recall,
            prc_thresholds,
        ) = torchmetrics.functional.classification.binary_precision_recall_curve(
            proba_pred, targets, thresholds=20
        )
        prc = curve_triplet_to_dict(
            (prc_precision, prc_recall, prc_thresholds),
            ("precision", "recall", "thresholds"),
        )
        auprc = (
            torchmetrics.functional.classification.binary_average_precision(
                proba_pred, targets, thresholds=None
            )
            .cpu()
            .item()
        )
        (
            roc_fpr,
            roc_tpr,
            roc_thresholds,
        ) = torchmetrics.functional.classification.binary_roc(
            proba_pred, targets, thresholds=20
        )
        roc = curve_triplet_to_dict(
            (roc_fpr, roc_tpr, roc_thresholds), ("fpr", "tpr", "thresholds")
        )
        auroc = (
            torchmetrics.functional.classification.binary_auroc(
                proba_pred, targets, thresholds=None
            )
            .cpu()
            .item()
        )
        ece = compute_ece(proba_pred, targets).cpu().item()
        score = (
            brier_score(get_one_hot_label(targets, num_classes=2), mean_proba)
            .cpu()
            .item()
        )
        conf_thresholds = torch.linspace(0, 10, 11) * 0.1
        metrics_list = ["acc", "precision", "recall", "f1", "mcc", "auprc", "auroc"]
        metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
            mean_proba, targets, thresholds=conf_thresholds, metrics_list=metrics_list
        )
        result_dict = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": acc,
            "mcc": mcc,
            "cmtx": cmtx,
            "prc": prc,
            "auprc": auprc,
            "roc": roc,
            "auroc": auroc,
            "ece": ece,
            "brier score": score,
        }
        for metric_name in metrics_list:
            result_dict[metric_name + "_list"] = [
                a.item() for a in metrics_dict[metric_name]
            ]
        result_dict["count_list"] = count_list

        ensemble_proba = torch.cat([p.transpose(0, 1) for p in ensemble_proba_list])
        confidence = torch.cat(confidence_list)
        entropy = compute_sampling_entropy(ensemble_proba)
        mutual_info = compute_sampling_mutual_information(ensemble_proba)
        # trunk-ignore(bandit/B101)
        assert (
            entropy.shape[0]
            == mutual_info.shape[0]
            == confidence.shape[0]
            == labels_pred.shape[0]
            == targets.shape[0]
        )
        uq_list: List[Union[Dict, None]] = [None] * targets.shape[0]
        for i in range(targets.shape[0]):
            if targets[i] == labels_pred[i]:
                if labels_pred[i] == 0:
                    quadrant = "tn"
                elif labels_pred[i] == 1:
                    quadrant = "tp"
                else:
                    raise ValueError(
                        "unexpected label {} at index {}".format(labels_pred[i], i)
                    )
            else:
                if labels_pred[i] == 0:
                    quadrant = "fn"
                elif labels_pred[i] == 1:
                    quadrant = "fp"
                else:
                    raise ValueError(
                        "unexpected label {} at index {}".format(labels_pred[i], i)
                    )

            uq_list[i] = {
                "confidence": confidence[i].cpu().item(),
                "target": targets[i].cpu().item(),
                "label_pred": labels_pred[i].cpu().item(),
                "entropy": entropy[i].cpu().item(),
                "muinfo": mutual_info[i].cpu().item(),
                "quadrant": quadrant,
            }
        result_dict["uq"] = uq_list
        return result_dict
