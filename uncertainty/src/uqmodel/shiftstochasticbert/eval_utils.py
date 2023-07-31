import logging
import numpy as np
import packaging
import torch
import torchmetrics
from typing import Callable, Dict, List, Tuple, Union
from uqmodel.shiftstochasticbert.stochastic_bert_mlc import StochasticBertBinaryClassifier
from uqmodel.shiftstochasticbert.experiment import ExperimentConfig
from uqmodel.shiftstochasticbert.ensemble_bert import EnsembleBertClassifier
from uqmodel.shiftstochasticbert.dropout_bert import DropoutBertClassifier
from uqmodel.shiftstochasticbert.sampling_metrics import (
    compute_sampling_entropy,
    compute_sampling_mutual_information
)


logger = logging.getLogger(__name__)

def set_zero_to_nextafter(proba:torch.Tensor) -> torch.Tensor:
    proba = proba.clone()
    proba[proba == 0] = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
    return proba

def compute_ece(proba, targets):
    if packaging.version.parse(torchmetrics.__version__) > packaging.version.parse('0.11.4'):
        ece = torchmetrics.functional.classification.binary_calibration_error(proba, targets, n_bins=15, norm='l1')
    else:
        proba = set_zero_to_nextafter(proba)
        ece = torchmetrics.functional.classification.binary_calibration_error(proba, targets, n_bins=15, norm='l1')
    return ece

def brier_score(targets, probs):
    sum = torch.sum((targets - probs)**2)
    score = sum / len(targets)
    return score

def curve_triplet_to_dict(xyt, keys):
    curve = dict()
    for v,k in zip(xyt, keys, strict=True):
        v = v.cpu().tolist() if v.get_device() >= 0 else v.tolist()
        curve[k] = v
    return curve

def get_one_hot_label(labels, num_classes=None):
    if num_classes:
        return torch.nn.functional.one_hot(labels, num_classes)
    else:
        return torch.nn.functional.one_hot(labels)

def compute_binary_metrics_vs_conf_from_tensors(
        test_proba_pred:torch.Tensor,
        test_label:torch.Tensor,
        thresholds:torch.Tensor=None,
        empty_is:float=1.,
        metrics_list:list=('acc')
    ) -> Tuple[Dict, List]:
    if not thresholds:
        thresholds = torch.linspace(0, 10, 11)*0.1
    metrics_func:Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
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
    metrics_dict:Dict = dict()
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

class EnsembleBertClassifierEvalautor(object):
    """Evaluate Bert classifier."""
    def __init__(self,
                 config:ExperimentConfig,
                 ensemble:EnsembleBertClassifier,
                 dataset:torch.utils.data.TensorDataset):
        self.config = config
        self.ensemble = ensemble
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      config.trainer.batch_size)

    def predict_proba(self,
                      dataloader:torch.utils.data.DataLoader,
                      device:Union[torch.device, str, None]):
        n_batches = len(dataloader)
        ensemble_proba_list, mean_proba_list, confidence_list, labels_list = [], [], [], []
        ensemble_model = self.ensemble.to(device)
        ensemble_model.eval()
        with torch.no_grad():
            for batch_idx,batch in enumerate(dataloader):
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                ensemble_proba, mean_proba, confidence, labels = ensemble_model.predict(input_ids, attention_mask)
                ensemble_proba_list.append(ensemble_proba)
                mean_proba_list.append(mean_proba)
                confidence_list.append(confidence)
                labels_list.append(labels)
                logger.debug('completed predict_proba for batch {} of {}'.format(batch_idx, n_batches))
        ensemble_model.train()
        return ensemble_proba_list, mean_proba_list, confidence_list, labels_list

    def get_targets(self, dataloader:torch.utils.data.DataLoader) -> List:
        targets = [t for _,_,t in dataloader]
        return targets

    def compute_eval_metrics(self, device:Union[torch.device, str, None]=None) -> dict:
        """Compute evaluaton metrics.

        compute evaluaton metrics including
           predictive metrics: precision, recall, accuracy, F1, AUROC, AUPR, MCC
           UQ metrics: ECE, Brier score, entropy (total uncertainty), data uncertainty,
                       mutual information (epistemic)
        """
        dataloader = self.dataloader
        logger.debug('predicting proba and confidence ...')
        ensemble_proba_list, mean_proba_list, confidence_list, labels_list = self.predict_proba(dataloader, device)
        logger.debug('computing predictive performance ...')
        mean_proba = torch.cat(mean_proba_list)
        proba_pred = mean_proba[:, 1].to(device)
        labels_pred = torch.cat(labels_list).to(device)
        targets_list = self.get_targets(dataloader)
        targets = torch.cat(targets_list).to(device)
        precision = torchmetrics.functional.classification.binary_precision(labels_pred, targets).cpu().item()
        recall = torchmetrics.functional.classification.binary_recall(labels_pred, targets).cpu().item()
        f1 = torchmetrics.functional.classification.binary_f1_score(labels_pred, targets).cpu().item()
        acc = torchmetrics.functional.classification.binary_accuracy(labels_pred, targets).cpu().item()
        mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(labels_pred, targets).cpu().item()
        cmtx = torchmetrics.functional.classification.binary_confusion_matrix(labels_pred, targets).cpu().tolist()
        prc_precision, prc_recall, prc_thresholds \
            = torchmetrics.functional.classification.binary_precision_recall_curve(
                proba_pred, targets, thresholds=20)
        prc = curve_triplet_to_dict((prc_precision, prc_recall, prc_thresholds), ('precision', 'recall', 'thresholds'))
        auprc = torchmetrics.functional.classification.binary_average_precision(
            proba_pred, targets, thresholds=None).cpu().item()
        roc_fpr, roc_tpr, roc_thresholds \
            = torchmetrics.functional.classification.binary_roc(
                proba_pred, targets, thresholds=20)
        roc = curve_triplet_to_dict((roc_fpr, roc_tpr, roc_thresholds), ('fpr', 'tpr', 'thresholds'))
        auroc = torchmetrics.functional.classification.binary_auroc(
            proba_pred, targets, thresholds=None).cpu().item()
        ece = compute_ece(proba_pred, targets).cpu().item()
        score = brier_score(get_one_hot_label(targets, num_classes=2), mean_proba).cpu().item()
        conf_thresholds = torch.linspace(0, 10, 11)*0.1
        metrics_list=['acc', 'precision', 'recall', 'f1', 'mcc', 'auprc', 'auroc']
        metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
            mean_proba, targets, thresholds=conf_thresholds, metrics_list=metrics_list)
        result_dict =  {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': acc,
            'mcc': mcc,
            'cmtx': cmtx,
            'prc': prc,
            'auprc': auprc,
            'roc': roc,
            'auroc': auroc,
            'ece': ece,
            'brier score': score,
            'sigma': self. config.datashift.sigma
        }
        for metric_name in metrics_list:
            result_dict[metric_name + '_list'] = [a.item() for a in metrics_dict[metric_name]]
        result_dict['count_list'] = count_list

        logger.debug('computing UQ measures ...')
        ensemble_proba = torch.cat([p.transpose(0, 1) for p in ensemble_proba_list])
        confidence = torch.cat(confidence_list)
        entropy = compute_sampling_entropy(ensemble_proba)
        mutual_info = compute_sampling_mutual_information(ensemble_proba)
        assert (entropy.shape[0]
                == mutual_info.shape[0]
                == confidence.shape[0]
                == labels_pred.shape[0]
                == targets.shape[0])
        uq_list:List[Union[Dict, None]] = [None]*targets.shape[0]
        for i in range(targets.shape[0]):
            if targets[i] == labels_pred[i]:
                if labels_pred[i] == 0:
                    quadrant = 'tn'
                elif labels_pred[i] == 1:
                    quadrant = 'tp'
                else:
                    raise ValueError('unexpected label {} at index {}'.format(labels_pred[i], i))
            else:
                if labels_pred[i] == 0:
                    quadrant = 'fn'
                elif labels_pred[i] == 1:
                    quadrant = 'fp'
                else:
                    raise ValueError('unexpected label {} at index {}'.format(labels_pred[i], i))

            uq_list[i] = {
                'confidence': confidence[i].cpu().item(),
                'target': targets[i].cpu().item(),
                'label_pred': labels_pred[i].cpu().item(),
                'entropy': entropy[i].cpu().item(),
                'muinfo': mutual_info[i].cpu().item(),
                'quadrant': quadrant
            }
        result_dict['uq'] = uq_list
        logger.debug('completed computing prediction and UQ measures ...')
        return result_dict

class DropoutBertClassifierEvalautor(object):
    """Evaluate Bert classifier."""
    def __init__(self,
                 config:ExperimentConfig,
                 dropout_model:DropoutBertClassifier,
                 dataset:torch.utils.data.TensorDataset):
        self.config = config
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      config.trainer.batch_size)
        self.dropout_model = dropout_model

    def predict_proba(self,
                      dataloader:torch.utils.data.DataLoader,
                      device:Union[torch.device, str, None]):
        n_batches = len(dataloader)
        dropout_proba_list, mean_proba_list, confidence_list, labels_list = [], [], [], []
        dropout_model = self.dropout_model.to(device)
        #  no need to switch to eval
        with torch.no_grad():
            for batch_idx,batch in enumerate(dataloader):
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                dropout_proba, mean_proba, confidence, labels = dropout_model.predict(input_ids, attention_mask)
                dropout_proba_list.append(dropout_proba)
                mean_proba_list.append(mean_proba)
                confidence_list.append(confidence)
                labels_list.append(labels)
                logger.debug('completed predict_proba for batch {} of {}'.format(batch_idx, n_batches))
        # no need to switch back to train
        return dropout_proba_list, mean_proba_list, confidence_list, labels_list

    def get_targets(self, dataloader:torch.utils.data.DataLoader) -> List:
        targets = [t for _,_,t in dataloader]
        return targets

    def compute_eval_metrics(self, device:Union[torch.device, str, None]=None) -> dict:
        """Compute evaluaton metrics.

        compute evaluaton metrics including
           predictive metrics: precision, recall, accuracy, F1, AUROC, AUPR, MCC
           UQ metrics: ECE, Brier score, entropy (total uncertainty), data uncertainty,
                       mutual information (epistemic)
        """
        dataloader = self.dataloader
        logger.debug('predicting proba and confidence ...')
        sampling_proba_list, mean_proba_list, confidence_list, labels_list = self.predict_proba(dataloader, device)
        logger.debug('computing predictive performance ...')
        mean_proba = torch.cat(mean_proba_list)
        proba_pred = mean_proba[:, 1].to(device)
        labels_pred = torch.cat(labels_list).to(device)
        targets_list = self.get_targets(dataloader)
        targets = torch.cat(targets_list).to(device)
        precision = torchmetrics.functional.classification.binary_precision(labels_pred, targets).cpu().item()
        recall = torchmetrics.functional.classification.binary_recall(labels_pred, targets).cpu().item()
        f1 = torchmetrics.functional.classification.binary_f1_score(labels_pred, targets).cpu().item()
        acc = torchmetrics.functional.classification.binary_accuracy(labels_pred, targets).cpu().item()
        mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(labels_pred, targets).cpu().item()
        cmtx = torchmetrics.functional.classification.binary_confusion_matrix(labels_pred, targets).cpu().tolist()
        prc_precision, prc_recall, prc_thresholds \
            = torchmetrics.functional.classification.binary_precision_recall_curve(
                proba_pred, targets, thresholds=20)
        prc = curve_triplet_to_dict((prc_precision, prc_recall, prc_thresholds), ('precision', 'recall', 'thresholds'))
        auprc = torchmetrics.functional.classification.binary_average_precision(
            proba_pred, targets, thresholds=None).cpu().item()
        roc_fpr, roc_tpr, roc_thresholds \
            = torchmetrics.functional.classification.binary_roc(
                proba_pred, targets, thresholds=20)
        roc = curve_triplet_to_dict((roc_fpr, roc_tpr, roc_thresholds), ('fpr', 'tpr', 'thresholds'))
        auroc = torchmetrics.functional.classification.binary_auroc(
            proba_pred, targets, thresholds=None).cpu().item()
        ece = compute_ece(proba_pred, targets).cpu().item()
        score = brier_score(get_one_hot_label(targets, num_classes=2), mean_proba).cpu().item()
        conf_thresholds = torch.linspace(0, 10, 11)*0.1
        metrics_list=['acc', 'precision', 'recall', 'f1', 'mcc', 'auprc', 'auroc']
        metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
            mean_proba, targets, thresholds=conf_thresholds, metrics_list=metrics_list)
        result_dict =  {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': acc,
            'mcc': mcc,
            'cmtx': cmtx,
            'prc': prc,
            'auprc': auprc,
            'roc': roc,
            'auroc': auroc,
            'ece': ece,
            'brier score': score,
            'sigma': self. config.datashift.sigma
        }
        for metric_name in metrics_list:
            result_dict[metric_name + '_list'] = [a.item() for a in metrics_dict[metric_name]]
        result_dict['count_list'] = count_list

        logger.debug('computing UQ measures ...')
        ensemble_proba = torch.cat([p.transpose(0, 1) for p in sampling_proba_list])
        confidence = torch.cat(confidence_list)
        entropy = compute_sampling_entropy(ensemble_proba)
        mutual_info = compute_sampling_mutual_information(ensemble_proba)
        assert (entropy.shape[0]
                == mutual_info.shape[0]
                == confidence.shape[0]
                == labels_pred.shape[0]
                == targets.shape[0])
        uq_list:List[Union[Dict, None]] = [None]*targets.shape[0]
        for i in range(targets.shape[0]):
            if targets[i] == labels_pred[i]:
                if labels_pred[i] == 0:
                    quadrant = 'tn'
                elif labels_pred[i] == 1:
                    quadrant = 'tp'
                else:
                    raise ValueError('unexpected label {} at index {}'.format(labels_pred[i], i))
            else:
                if labels_pred[i] == 0:
                    quadrant = 'fn'
                elif labels_pred[i] == 1:
                    quadrant = 'fp'
                else:
                    raise ValueError('unexpected label {} at index {}'.format(labels_pred[i], i))

            uq_list[i] = {
                'confidence': confidence[i].cpu().item(),
                'target': targets[i].cpu().item(),
                'label_pred': labels_pred[i].cpu().item(),
                'entropy': entropy[i].cpu().item(),
                'muinfo': mutual_info[i].cpu().item(),
                'quadrant': quadrant
            }
        result_dict['uq'] = uq_list
        logger.debug('completed computing prediction and UQ measures ...')
        return result_dict


class VanillaBertClassifierEvalautor(object):
    """Evaluate Bert classifier."""
    def __init__(self,
                 config:ExperimentConfig,
                 model:StochasticBertBinaryClassifier,
                 dataset:torch.utils.data.TensorDataset):
        self.config = config
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      config.trainer.batch_size)
        self.model = model

    def predict_proba(self,
                      dataloader:torch.utils.data.DataLoader,
                      device:Union[torch.device, str, None]):
        n_batches = len(dataloader)
        proba_list, confidence_list, labels_list = [], [], []
        model = self.model.to(device)
        model.eval()
        with torch.no_grad():
            for batch_idx,batch in enumerate(dataloader):
                input_ids, attention_mask, _ = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                _, proba, confidence, labels = model.predict(input_ids, attention_mask)
                proba_list.append(proba)
                confidence_list.append(confidence)
                labels_list.append(labels)
                logger.debug('completed predict_proba for batch {} of {}'.format(batch_idx, n_batches))
        model.train()
        return proba_list, confidence_list, labels_list

    def get_targets(self, dataloader:torch.utils.data.DataLoader) -> List:
        targets = [t for _,_,t in dataloader]
        return targets

    def compute_eval_metrics(self, device:Union[torch.device, str, None]=None) -> dict:
        """Compute evaluaton metrics.

        compute evaluaton metrics including
           predictive metrics: precision, recall, accuracy, F1, AUROC, AUPR, MCC
           UQ metrics: ECE, Brier score, entropy (total uncertainty), data uncertainty,
                       mutual information (epistemic)
        """
        dataloader = self.dataloader
        logger.debug('predicting proba and confidence ...')
        proba_list, confidence_list, labels_list = self.predict_proba(dataloader, device)
        logger.debug('computing predictive performance ...')
        all_proba = torch.cat(proba_list)
        proba_pred = all_proba[:, 1].to(device)
        labels_pred = torch.cat(labels_list).to(device)
        targets_list = self.get_targets(dataloader)
        targets = torch.cat(targets_list).to(device)
        precision = torchmetrics.functional.classification.binary_precision(labels_pred, targets).cpu().item()
        recall = torchmetrics.functional.classification.binary_recall(labels_pred, targets).cpu().item()
        f1 = torchmetrics.functional.classification.binary_f1_score(labels_pred, targets).cpu().item()
        acc = torchmetrics.functional.classification.binary_accuracy(labels_pred, targets).cpu().item()
        mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(labels_pred, targets).cpu().item()
        cmtx = torchmetrics.functional.classification.binary_confusion_matrix(labels_pred, targets).cpu().tolist()
        prc_precision, prc_recall, prc_thresholds \
            = torchmetrics.functional.classification.binary_precision_recall_curve(
                proba_pred, targets, thresholds=20)
        prc = curve_triplet_to_dict((prc_precision, prc_recall, prc_thresholds), ('precision', 'recall', 'thresholds'))
        auprc = torchmetrics.functional.classification.binary_average_precision(
            proba_pred, targets, thresholds=None).cpu().item()
        roc_fpr, roc_tpr, roc_thresholds \
            = torchmetrics.functional.classification.binary_roc(
                proba_pred, targets, thresholds=20)
        roc = curve_triplet_to_dict((roc_fpr, roc_tpr, roc_thresholds), ('fpr', 'tpr', 'thresholds'))
        auroc = torchmetrics.functional.classification.binary_auroc(
            proba_pred, targets, thresholds=None).cpu().item()
        ece = compute_ece(proba_pred, targets).cpu().item()
        score = brier_score(get_one_hot_label(targets, num_classes=2), all_proba).cpu().item()
        conf_thresholds = torch.linspace(0, 10, 11)*0.1
        metrics_list=['acc', 'precision', 'recall', 'f1', 'mcc', 'auprc', 'auroc']
        metrics_dict, count_list = compute_binary_metrics_vs_conf_from_tensors(
            all_proba, targets, thresholds=conf_thresholds, metrics_list=metrics_list)
        result_dict =  {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'acc': acc,
            'mcc': mcc,
            'cmtx': cmtx,
            'prc': prc,
            'auprc': auprc,
            'roc': roc,
            'auroc': auroc,
            'ece': ece,
            'brier score': score,
            'sigma': self. config.datashift.sigma
        }
        for metric_name in metrics_list:
            result_dict[metric_name + '_list'] = [a.item() for a in metrics_dict[metric_name]]
        result_dict['count_list'] = count_list
        return result_dict

class EnsembleModelSelector(object):
    def __init__(self,
                 ensemble:EnsembleBertClassifier,
                 dataset:torch.utils.data.TensorDataset,
                 config:ExperimentConfig):
        self.ensemble = ensemble
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      config.trainer.batch_size)
        self.config = config

    def _predict_class_by_individual(self, model_idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_list, targets_list = [], []
        n_batches = len(self.dataloader)
        self.ensemble.to(self.config.device)
        for idx,batch in enumerate(self.dataloader):
            logger.debug('computing proba for batch {} of {} for model {}'.format(
                idx, n_batches, model_idx
            ))
            input_ids, attention_mask, target_labels = batch
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            target_labels = target_labels.to(self.config.device)
            proba = self.ensemble[model_idx].predict_proba(input_ids, attention_mask)
            predicted_labels = torch.argmax(proba, dim=1)
            predicted_list.append(predicted_labels)
            targets_list.append(target_labels)
        predicted_labels = torch.cat(predicted_list, dim=0)
        target_labels = torch.cat(targets_list, dim=0)
        return predicted_labels, target_labels


    def select_member_model(self,
                            selection_critieria='best_f1'
                            ) -> StochasticBertBinaryClassifier:
        if selection_critieria == 'random':
            idx = np.random.randint(0, high=self.__len__(), dtype=int)
            return self.model_ensemble[idx]

        scores = np.zeros(len(self.ensemble))
        for i in range(len(self.ensemble)):
            if selection_critieria in ['best_f1', 'median_f1']:
                logger.debug('predicting for member model {}'.format(i))
                predicted_labels, target_labels = self._predict_class_by_individual(i)
                f1 = torchmetrics.functional.classification.binary_f1_score(
                    predicted_labels,
                    target_labels)
                scores[i] = f1
            else:
                raise ValueError('unsupported selection_criteria {}'.format(selection_critieria))
        if selection_critieria == 'best_f1':
            idx = np.argmax(scores)
        elif selection_critieria == 'median_f1':
            idx = scores.tolist().index(np.percentile(scores, 50, interpolation='nearest'))
        elif selection_critieria == 'random':
            pass
        else:
            raise ValueError('unsupported member model selection method {}'.format(selection_critieria))
        logger.info('selected member model {} with scores {}'.format(idx, scores))
        self.ensemble.to(torch.device("cpu"))
        return self.ensemble[idx]
