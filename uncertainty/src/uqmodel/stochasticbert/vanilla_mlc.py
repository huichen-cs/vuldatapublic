import torch
from typing import Generator

class VanillaClassifier(object):
    def __init__(self, model, outputs:int=2):
        self.model = model
        self.n_outputs = outputs

    def predict_logits(self, data_loader:torch.utils.data.DataLoader):
        if next(self.model.parameters()).get_device() >=0:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        batch_list = []
        with torch.no_grad():
            for batch in data_loader:
                x, _ = batch
                x = x.to(device)
                self.model.eval()
                logits = self.model(x)
                self.model.train()
                batch_list.append(logits)
        return torch.cat(batch_list, dim=0)

    def predict_logits_proba(self, data_loader:torch.utils.data.DataLoader):
        logits = self.predict_logits(data_loader)
        proba = torch.softmax(logits, dim=-1)
        return logits, proba


    def predict_proba(self, test_dataloader, device=None):
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                self.model.eval()
                logits = self.model(x)
                proba_batch = torch.softmax(logits, dim=1)
                self.model.train()
                yield proba_batch

    def predict_class(self, test_dataloader, device=None):
        for proba_batch in self.predict_proba(test_dataloader, device):
            yield torch.argmax(proba_batch, dim=1)

    def predict_class_with_conf(self, test_dataloader, device=None):
        for proba_batch in self.predict_proba(test_dataloader, device):
            yield torch.max(proba_batch, dim=1)

    @classmethod
    def compute_class_with_conf(self, proba_generator:Generator[torch.Tensor, None, None], device=None):
        for proba_batch in proba_generator:
            proba, labels = torch.max(proba_batch, dim=1)
            yield proba, labels

