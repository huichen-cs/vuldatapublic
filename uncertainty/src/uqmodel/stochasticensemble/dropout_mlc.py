import torch

class DropoutClassifier(object):
    def __init__(self, model, outputs:int=2):
        self.model = model
        self.n_outputs = outputs

    def predict_logits(self, data_loader:torch.utils.data.DataLoader, n_stochastic_passes:int):
        if next(self.model.parameters()).get_device() >=0:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        batch_list = []
        with torch.no_grad():
            for batch in data_loader:
                x, _ = batch
                batch_size, output_size = x.shape[0], self.n_outputs
                batch_logits = torch.zeros(n_stochastic_passes, batch_size, output_size).to(device)
                x = x.to(device)
                for sample_idx in range(n_stochastic_passes):
                    self.model.train() # set train model for stochastic samples
                    logits = self.model(x)
                    batch_logits[sample_idx, :, :] = logits
                batch_list.append(batch_logits.transpose(1, 0))
        return torch.cat(batch_list, dim=0)

    def predict_logits_proba(self, data_loader:torch.utils.data.DataLoader, n_stochastic_passes):
        logits = self.predict_logits(data_loader, n_stochastic_passes)
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

    def predict_class_from_proba(self, test_dataloader, device=None):
        for proba_batch in self.predict_proba(test_dataloader, device):
            yield torch.argmax(proba_batch, dim=1)

    def predict_from_dropouts(self, test_dataloader, n_samples=1000, device=None):
        with torch.no_grad():
            for test_batch in test_dataloader:
                x, _ = test_batch
                if device is not None:
                    x = x.to(device)
                self.model.eval()
                proba_batch = self.model(x)
                predicted_class_batch = torch.argmax(proba_batch, dim=1)
                self.model.train()
                samples_logits = []
                samples_proba = []
                for _ in range(n_samples):
                    logits = self.model(x)
                    proba = torch.nn.functional.softmax(logits, dim=1)
                    samples_logits.append(logits)
                    samples_proba.append(proba)
                samples_logits_batch = torch.stack(samples_logits, dim=1).transpose(0, 1)
                samples_proba_batch = torch.stack(samples_proba, dim=1).transpose(0, 1)
                proba_samples_mean_batch = torch.mean(samples_proba_batch, dim=0)
                predicted_class_samples_proba, predicted_class_samples_batch = torch.max(proba_samples_mean_batch, dim=1)
                yield predicted_class_batch, proba_batch, \
                    samples_logits_batch, samples_proba_batch, predicted_class_samples_batch, predicted_class_samples_proba, \
                    proba_samples_mean_batch
