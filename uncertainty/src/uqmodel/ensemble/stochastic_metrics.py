import torch

def softmax_instance(mu, sigma_log_sigma, n_samples, passed_log_sigma=True):
    if passed_log_sigma:
        sigma = torch.exp(sigma_log_sigma)
    else:
        sigma = sigma_log_sigma
    if mu.get_device() >= 0:
        samples = torch.randn(n_samples, sigma.shape[0]).cuda()
    else:
        samples = torch.randn(n_samples, sigma.shape[0])
    logits = (
        mu.unsqueeze(0).repeat(n_samples, 1) +
        sigma.unsqueeze(0).repeat(n_samples, 1) * samples
    )
    proba = torch.softmax(logits, dim=1)
    std, proba = torch.std_mean(proba, dim=0)
    return std, proba

def softmax_batch(mu, sigma_log_sigma, n_samples, passed_log_sigma=True, return_mean_std=True):
    batch_size, output_size = mu.shape

    # compute z = mu + sigma epislon, where epislon ~ N(0, 1)
    if passed_log_sigma:
        sigma = torch.exp(sigma_log_sigma)
    else:
        sigma = sigma_log_sigma
    if mu.get_device() >= 0:
        samples = torch.randn(n_samples, batch_size, output_size).cuda()
    else:
        samples = torch.randn(n_samples, batch_size, output_size)
    logits_samples = (
            mu.unsqueeze(0).repeat(n_samples, 1, 1) +       # mu repeated for n_samples times
            torch.mul(
                sigma.unsqueeze(0).repeat(n_samples, 1, 1), # sigma repeated for n_samples times
                samples                                     # samples from standard normal distribution
            )
    )
    proba = torch.softmax(logits_samples, dim=-1)
    std, proba = torch.std_mean(proba, dim=0)
    if return_mean_std:
        return torch.mean(std, dim=-1), proba
    else:
        return std, proba


def entropy_instance(proba):
    log_proba = torch.log2(proba)
    p_log_p = log_proba * proba
    entropy = - p_log_p.mean()
    return entropy


# def entropy_batch(proba):
#     log_proba = torch.log2(proba)
#     p_log_p = torch.mul(log_proba, proba)
#     entropy = - p_log_p.mean(dim=-1)
#     return entropy

def entropy_batch(proba):
    log_proba = torch.log2(proba)
    p_log_p = torch.mul(log_proba, proba)
    entropy = - p_log_p.sum(dim=-1)
    return entropy

def softmax_all(mu_all, sigma_log_sigma_all, n_samples, passed_log_sigma=True, return_mean_std=True):
    mu_all = mu_all.transpose(0, -1).transpose(1, -1)
    sigma_all = sigma_log_sigma_all.transpose(0, -1).transpose(1, -1)
    # trunk-ignore(bandit/B101)
    assert mu_all.shape == sigma_all.shape
    proba_list = []
    for i in range(len(mu_all)):
        mu = mu_all[i]
        sigma = sigma_all[i]

        batch_size, output_size = mu.shape
        if passed_log_sigma:
            sigma = torch.exp(sigma)
        if mu.get_device() >= 0:
            samples = torch.randn(n_samples, batch_size, output_size).cuda()
        else:
            samples = torch.randn(n_samples, batch_size, output_size)
        logits_samples = (
                mu.unsqueeze(0).repeat(n_samples, 1, 1) +       # mu repeated for n_samples times
                torch.mul(
                    sigma.unsqueeze(0).repeat(n_samples, 1, 1), # sigma repeated for n_samples times
                    samples                                     # samples from standard normal distribution
                )
        )
        proba = torch.softmax(logits_samples, dim=-1)
        proba_list.append(proba)
    proba_all = torch.cat(proba_list)
    proba_all = proba_all.transpose(0, 1)
    std = torch.std(proba_all, dim=1)
    if return_mean_std:
        return torch.mean(std, dim=-1), proba_all
    else:
        return std, proba_all
