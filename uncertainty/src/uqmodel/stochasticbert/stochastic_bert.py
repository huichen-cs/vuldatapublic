import collections
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel
from uqmodel.stochasticbert.loss import StochasticCrossEntropyLoss

class StochasticBertBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.1, cache_dir='~/.hfcache'):
        super().__init__()
        cache_dir = os.path.expanduser(cache_dir)
        self.num_classes = num_classes

        output_mu_dict = collections.OrderedDict()
        output_sigma_dict = collections.OrderedDict()
        
        # shared bert module
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base", cache_dir=cache_dir)
    
        # dual-head output module with dropouts
        # head 1. logits for mu
        output_mu_dict['dropout_mu'] = torch.nn.Dropout(dropout_prob)
        output_mu_dict['output_mu'] = torch.nn.Linear(self.bert.config.hidden_size, self.num_classes)
        # head 2. output for sigma
        output_sigma_dict['dropout_sigma'] = torch.nn.Dropout(dropout_prob)
        output_sigma_dict['output_sigma'] = torch.nn.Linear(self.bert.config.hidden_size, self.num_classes)
        output_sigma_dict['activation_sigma'] = torch.nn.Softplus()

        # assemble the model
        self.mu_module =  torch.nn.Sequential(output_mu_dict)
        self.sigma_module = torch.nn.Sequential(output_sigma_dict)

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # pass it to mu
        mu = self.mu_module(pooled_output)
        # pass it to sigma
        sigma = self.sigma_module(pooled_output)
        return mu, sigma
    
    def _output_shape(self):
        return (self.bert.config.hidden_size, self.num_classes)
    
    @property
    def output_shape(self):
        return self._output_shape()

class StochasticBertBinaryClassifierTrainer(object):
    def __init__(self, model, train_loader, optimizer, loss_fn, device=None, evaluator=None, disable_tqdm=False):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        if not isinstance(loss_fn, StochasticCrossEntropyLoss):
            raise ValueError('loss_fn must be a StochasticCrossEntropyLoss instance')
        else:
            self.loss_fn = loss_fn
        self.device = device
        self.evaluator = evaluator
        self.disable_tqdm = disable_tqdm

    def step(self):
        self.model.train()
        self.model.to(self.device)
        total_loss = 0
        for input_ids, attention_mask, targets in tqdm(self.train_loader, desc='train_step', disable=self.disable_tqdm):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(logits, targets)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs), desc='train', disable=self.disable_tqdm):
            train_loss = self.step()
            accuracy = self.evaluator.evaluate(self.train_loader)
            print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f} Accuracy: {accuracy: .4f}")

class StochasticBertBinaryClassifierEvaluator(object):
    @classmethod
    def batch_stochastic_softmax(self, mu, sigma_log_sigma, n_samples, passed_log_sigma=True, return_mean_std=True):
        batch_size, output_size = mu.shape[0], mu.shape[1:]

        # compute z = mu + sigma epislon, where epislon ~ N(0, 1)
        if passed_log_sigma:
            sigma = torch.exp(sigma_log_sigma)
        else:
            sigma = sigma_log_sigma
        if mu.get_device() >= 0:
            samples = torch.randn(n_samples, batch_size, *output_size).cuda()
        else:
            samples = torch.randn(n_samples, batch_size, *output_size)
        logits_samples = (
                mu.unsqueeze(0).repeat(n_samples, *([1]*len(mu.shape))) +       # mu repeated for n_samples times
                torch.mul(
                    sigma.unsqueeze(0).repeat(n_samples, *([1]*len(sigma.shape))), # sigma repeated for n_samples times
                    samples                                     # samples from standard normal distribution
                )
        )
        proba = torch.softmax(logits_samples, dim=-1)
        std, proba = torch.std_mean(proba, dim=0)
        if return_mean_std:
            return torch.mean(std, dim=-1), proba
        else:
            return std, proba
    
    def __init__(self,
                 model,
                 n_stochastic_passes,
                 n_aleatoric_samples,
                 device,
                 disable_tqdm=False):
        self.model = model
        self.n_stochastic_passes = n_stochastic_passes # default 1000?
        self.n_aleatoric_samples = n_aleatoric_samples # default 100
        self.device = device
        self.disable_tqdm = disable_tqdm

    def batch_stochastic_eval_step(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Compute the output logits with dropouts being on
        self.model.train()
        with torch.no_grad():
            mu, sigma = self.model(input_ids, attention_mask)
        return mu, sigma  

    def batch_stochastic_pass(self, input_ids, attention_mask):
        mu_samples = torch.zeros(self.n_stochastic_passes, len(input_ids), self.model.output_shape[-1])
        sigma_samples = torch.zeros(self.n_stochastic_passes, len(input_ids), self.model.output_shape[-1])
        for i in tqdm(range(self.n_stochastic_passes), desc='dropout_passes', disable=self.disable_tqdm):
            mu, sigma = self.batch_stochastic_eval_step(input_ids, attention_mask)
            mu_samples[i] = mu
            sigma_samples[i] = sigma
        return mu_samples, sigma_samples


    def evaluate(self, test_loader):
        # Initialize variables for computing metrics
        total_correct = 0
        total_samples = 0
    
        # Loop over the test data
        for batch in tqdm(test_loader, desc='eval', disable=self.disable_tqdm):
            input_ids, attention_mask, targets = batch
            mu_samples, sigma_samples = self.batch_stochastic_pass(input_ids, attention_mask)
            proba_passes = torch.zeros(self.n_stochastic_passes, len(input_ids), *(self.model.output_shape[1:]))
            for i,(mu,sigma) in tqdm(enumerate(zip(mu_samples,
                                                   sigma_samples,
                                                   strict=True)), desc='eval_softmax', disable=self.disable_tqdm):
                _, proba = self.batch_stochastic_softmax(mu,
                                                         sigma,
                                                         self.n_aleatoric_samples,
                                                         passed_log_sigma=False,
                                                         return_mean_std=True)
                proba_passes[i] = proba
            proba_mean = torch.mean(proba_passes, dim=0)      
            preds = torch.argmax(proba_mean, dim=-1)
    
            # Update metrics
            total_correct += (targets == preds).sum().item()
            total_samples += len(targets)
    
        # Compute evaluation metrics
        accuracy = total_correct / total_samples
        # precision = ...
        # recall = ...
        # f1_score = ...
    
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Precision: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1 score: {f1_score:.4f}")
        return accuracy
        
