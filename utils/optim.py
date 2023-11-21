import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_optimizer(type, lr, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if type == "sgd":
        optimizer = SGD(parameters, lr=lr)
    elif type == "adam":
        optimizer = Adam(parameters, lr=lr, eps=eps)
    elif type == "adamw":
        optimizer = AdamW(parameters, lr=lr)
    else:
        raise ValueError("optimizer not recognized!")
    return optimizer


def get_scheduler(type, lr, lr_end, max_step, optimizer):
    eps = 1e-8
    if type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_step, eta_min=eps)
    else:
        scheduler_module = getattr(torch.optim.lr_scheduler, type)
        if lr_end:
            assert type == "ExponentialLR"
            gamma = (lr_end / lr) ** (1.0 / max_step)
        scheduler = scheduler_module(optimizer, gamma=gamma)
    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
