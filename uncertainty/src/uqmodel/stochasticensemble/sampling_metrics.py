import torch

def compute_sampling_entropy(class_proba:torch.Tensor):
    """
    Compute entroy from class probability samples.

    Parameters
    ----------
    class_proba :
        a tensor of n x n_samples x n_classes
    """
    mean_proba = torch.mean(class_proba, -2)
    log_mean_proba = torch.log2(mean_proba)
    proba_log_proba = torch.mul(mean_proba, log_mean_proba)
    entropy = - torch.sum(proba_log_proba, dim=-1)
    return entropy

def compute_sampling_mutual_information(class_proba:torch.Tensor):
    """
    Compute mutual information from class probability samples, mutual information
    is also called information gain.

    Parameters
    ----------
    class_proba :
        a tensor of n x n_samples x n_classes
    """
    entropy = compute_sampling_entropy(class_proba)
    
    log_proba = torch.log2(class_proba)
    proba_log_proba = torch.mul(class_proba, log_proba)
    proba_log_proba[class_proba == 0] = 0
    sample_entropy = - torch.sum(proba_log_proba, dim=-1)
    mean_sample_entropy = torch.mean(sample_entropy, -1)

    mutual_information = entropy - mean_sample_entropy
    return mutual_information
