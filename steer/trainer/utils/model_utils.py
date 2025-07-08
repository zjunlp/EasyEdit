import torch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
    return total_norm