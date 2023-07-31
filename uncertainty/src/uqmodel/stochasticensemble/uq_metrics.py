import torch
import torchmetrics
import typing

def brier_score(targets: typing.Iterable[torch.Tensor], probs: typing.Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Brier score.
    
    Reference:
        https://en.wikipedia.org/wiki/Brier_score
        https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    """
    sum = 0
    n = 0
    for t,p in zip(targets, probs, strict=True):
        sum += torch.sum((t - p)**2)
        n += len(t)
        # print(n, sum)
    return sum/n

def brier_score_from_tensors(targets, probs):
    sum = torch.sum((targets - probs)**2)
    score = sum / len(targets)
    return score

def compute_accuracy_from_labels(test_label_pred: typing.Iterable[torch.Tensor], test_label: typing.Iterable[torch.Tensor]) -> torch.Tensor:
    correct = 0
    total = 0
    for label_batch,pred_label_batch in zip(test_label,
                                            test_label_pred,
                                            strict=True):
        correct += torch.sum(label_batch == pred_label_batch)
        total += len(label_batch)
    return correct/total

def compute_accuracy(test_proba_pred: typing.Iterable[torch.Tensor], test_label: typing.Iterable[torch.Tensor]) -> torch.Tensor:
    test_proba_pred = torch.cat(test_proba_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    acc = torchmetrics.functional.classification.binary_accuracy(test_proba_pred, test_label)
    return acc

def compute_binary_metrics_vs_conf_from_tensors(test_proba_pred:torch.Tensor,
                                                test_label:torch.Tensor,
                                                thresholds=None,
                                                empty_is=1.,
                                                metrics_list:list=('acc')):
    if not thresholds:
        thresholds = torch.linspace(0, 10, 11)*0.1
    metrics_func = {
        'acc': torchmetrics.functional.classification.binary_accuracy,
        'f1': torchmetrics.functional.classification.binary_f1_score,
        'recall': torchmetrics.functional.classification.binary_recall,
        'precision': torchmetrics.functional.classification.binary_precision,
        'mcc': torchmetrics.functional.classification.binary_matthews_corrcoef,
        'auprc': torchmetrics.functional.classification.binary_average_precision,
        'auroc': torchmetrics.functional.classification.binary_auroc
    }
    if not metrics_list:
        metrics_list = ['acc']
    metrics_dict = dict()
    for metric_name in metrics_list:
        if metric_name not in metrics_func.keys():
            raise ValueError('metrics must be in {}'.format(metrics_func.keys()))
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

def compute_binary_acc_vs_conf_from_tensors(test_proba_pred:torch.Tensor,
                                                test_label:torch.Tensor,
                                                thresholds=None,
                                                empty_is=1.):
    if not thresholds:
        thresholds = torch.linspace(0, 10, 11)*0.1
    acc_list = []
    count_list = []
    for t in thresholds:
        pred = test_proba_pred[test_proba_pred[:, 1] >= t]
        test = test_label[test_proba_pred[:, 1] >= t]
        count = len(test)
        if count > 0:
            acc = torchmetrics.functional.classification.binary_accuracy(pred[:, 1], test)
        else:
            acc = torch.tensor(empty_is)
        acc_list.append(acc)
        count_list.append(count)
    return acc_list, count_list

def compute_binary_acc_vs_conf(test_proba_pred:typing.Iterable[torch.Tensor],
                               test_label: typing.Iterable[torch.Tensor],
                               thresholds=None,
                               empty_is=1.):
    if not thresholds:
        thresholds = torch.linspace(0, 10, 11)*0.1
    test_proba_pred = torch.cat(test_proba_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    acc_list, count_list = compute_binary_acc_vs_conf_from_tensors(test_proba_pred,
                                                                   test_label,
                                                                   thresholds=thresholds,
                                                                   empty_is=empty_is)
    return acc_list, count_list
