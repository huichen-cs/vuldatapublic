import os
import torch
from uqmodel.stochasticbert.stochastic_metrics import softmax_batch
from uqmodel.stochasticbert.stochastic_bert import StochasticBertBinaryClassifier

class StochasticEnsembleBertClassifier(object):
    """
    An ensemble of bert classifiers.
    
    """
    def __init__(self, ensemble_size:int, num_classes:int=2, dropout_prob:float=0.1,
                 cache_dir:str='~/.hfcache'):
        cache_dir = os.path.expanduser(cache_dir)
        self.size = ensemble_size
        self.model_ensemble = [StochasticBertBinaryClassifier(num_classes, dropout_prob, cache_dir) for _ in range(ensemble_size)]
        self.log_sigma = False

    # def to(self, device:torch.device):
    #     self.model_ensemble = [model.to(device) for model in self.model_ensemble]
    #     return self

    def __getitem__(self, idx:int):
        return self.model_ensemble[idx]
        
    def __len__(self):
        return len(self.model_ensemble)

    def predict_proba(self, test_dataloader:torch.utils.data.DataLoader, n_samples:int, device=None):
        # testing
        with torch.no_grad():
            for test_batch in test_dataloader:
                input_ids, attention_mask, _ = test_batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                proba_list = []

                for idx in range(self.size):
                    self.model_ensemble[idx].to(device)
                    self.model_ensemble[idx].eval()
                    mu, sigma = self.model_ensemble[idx](input_ids, attention_mask)
                    _, proba = softmax_batch(mu, sigma, n_samples, passed_log_sigma=self.log_sigma)
                    proba_list.append(proba)
                    self.model_ensemble[idx].train()
                yield proba_list

    def predict_mean_proba(self, test_dataloader:torch.utils.data.DataLoader, n_samples:int, device:torch.device=None):
        test_proba = self.predict_proba(test_dataloader, n_samples, device)
        for test_batch in test_proba:
            mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
            yield mean_proba

    # def predict(self, test_dataloader, device=None):
    #     test_proba = self.predict_proba(test_dataloader, device)
    #     for test_batch in test_proba:
    #         mean_proba = torch.stack(test_batch, dim=0).mean(dim=0)
    #         yield torch.argmax(mean_proba, dim=1)

    @classmethod
    def compute_class_with_conf(self, mean_proba_iterable):
        for mean_proba_batch in mean_proba_iterable:
            proba, labels = torch.max(mean_proba_batch, dim=-1)
            yield proba, labels
