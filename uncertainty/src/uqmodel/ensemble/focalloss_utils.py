import logging
import torch
from uqmodel.ensemble.ps_data import get_dataset_stats


logger = logging.getLogger('focalloss')

def compute_focalloss_alpha(train_dataset, n_classes, imbalance_ratio, device):
    if n_classes != 2:
        raise ValueError('implemented only for n_classes=2, unimplemented for n_classes={}'.format(n_classes))
    stats = get_dataset_stats(train_dataset)
    # trunk-ignore(bandit/B101)
    assert stats['n_rows_1']*imbalance_ratio == stats['n_rows_0']
    focal_alpha = torch.tensor([stats['n_rows']/(n_classes*stats['n_rows_0']), stats['n_rows']/(n_classes*stats['n_rows_1'])])
    logger.info('focal_alpha = {}'.format(focal_alpha))
    focal_alpha = focal_alpha.to(device)
    return focal_alpha
