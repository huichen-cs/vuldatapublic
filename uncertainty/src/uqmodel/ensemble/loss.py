import torch

class StochasticCrossEntropyLoss(torch.nn.Module):
    """
    Stochastic negative log-likelihood loss function.

    The loss function takes logits as input, following the convention of Pytorch, name this
    as cross entropy loss. Negative log-likelihood and cross-entropy loss are mathemtically
    equivalent.
    """
    def __init__(self, n_samples:int, use_log_sigma:bool):
        super().__init__()
        self.n_samples = n_samples
        self.use_log_sigma = use_log_sigma


    def forward(self, X, y):
        if self.use_log_sigma:
            mu, log_sigma = X
            sigma = torch.exp(log_sigma)
        else:
            mu, sigma = X
        batch_size, output_size = mu.shape

        # compute z = mu + sigma epislon, where epislon ~ N(0, 1)
        if mu.get_device() >= 0:
            samples = torch.randn(self.n_samples, batch_size, output_size).cuda()
        else:
            samples = torch.randn(self.n_samples, batch_size, output_size)
        logits_samples = (
                mu.unsqueeze(0).repeat(self.n_samples, 1, 1) +   # mu repeated for n_samples times
                torch.mul(
                    sigma.unsqueeze(0).repeat(self.n_samples, 1, 1), # sigma repeated for n_samples times
                    samples                                                # samples from standard normal distribution
                )
        )

        # L_x = - \sum\limits_i \log \frac{1}{T}
        #         \sum\limits_t e^{\hat{x}_{i, t, c} -
        #                          \log \sum\limits_{c^\prime} e^{\hat{x}_{x, t, c^\prime}}}
        # where
        #     $t$ indexes samples of predictive samples $\hat{x}_{i, t}$ that is a tensor
        #         whose last dimension is the number of classes $N_c$
        #     $hat{x}_{i, t, c}$ selects logits based on true class label.
        logits_predict = torch.logsumexp(logits_samples, dim=-1)
        logits_by_truth = logits_samples[:, torch.arange(logits_samples.shape[1]).type_as(y), y]
        loss = - torch.sum(torch.logsumexp(logits_by_truth - logits_predict, dim=0) - torch.log(torch.tensor(self.n_samples).float()))
        # loss = - torch.sum(
        #         torch.logsumexp(logits_by_truth - logits_predict, dim=0) - torch.log(torch.tensor(self.n_samples).float())
        #         -
        #         torch.log(self.n_samples)
        #     )
        return loss
