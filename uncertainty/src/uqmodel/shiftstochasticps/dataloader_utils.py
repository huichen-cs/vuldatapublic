import torch


def yield_one_hot_test_label(test_loader, device=None, num_class=None):
    for batch in test_loader:
        if num_class:
            yield torch.nn.functional.one_hot(batch[1], num_class).to(device)
        else:
            yield torch.nn.functional.one_hot(batch[1]).to(device)


def get_one_hot_test_label(test_loader, device=None, num_class=None):
    labels = list(yield_one_hot_test_label(test_loader, device, num_class))
    labels_one_hot = torch.cat(labels, dim=0)
    return labels_one_hot


def yield_test_label(test_loader, device=None):
    for batch in test_loader:
        # logger.info('label: {}'.format(batch[1]))
        yield batch[1].to(device)


def get_test_label(test_loader, device=None):
    tensor_list = []
    for batch in test_loader:
        tensor_list.append(batch[-1].to(device))
    return torch.cat(tensor_list, dim=0)
