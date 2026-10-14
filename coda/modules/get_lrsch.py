import torch
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from models.ops_misc import CosineWarmupLR

def get_optimizer(optim_name, model_params, optim_kwargs):
    """
    Factory function to create optimizer instance
    Args:
        optim_name: Name of the optimizer
        model_params: Model parameters to optimize
        optim_kwargs: Optimizer keyword arguments
    Returns:
        torch.optim.Optimizer instance
    """
    optimizers = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam, 
        'adamw': torch.optim.AdamW,
        'rmsprop': torch.optim.RMSprop
    }
    
    if optim_name not in optimizers:
        raise ValueError(f"Optimizer {optim_name} is not supported. Available optimizers: {list(optimizers.keys())}")
        
    optimizer_class = optimizers[optim_name]
    return optimizer_class(model_params, **optim_kwargs)

def get_scheduler(lrsch_name, optimizer, lrsch_kwargs, train_loader_len):
    if lrsch_name == 'step':
        lrsch = torch.optim.lr_scheduler.StepLR 
        if 'step_size' not in lrsch_kwargs:
            lrsch_kwargs['step_size'] = int(2.4 * train_loader_len)
    elif lrsch_name == 'multi_step':
        lrsch = torch.optim.lr_scheduler.MultiStepLR
    elif lrsch_name == 'cosine_annealing':
        lrsch = torch.optim.lr_scheduler.CosineAnnealingLR
        lrsch_kwargs['eta_min'] = float(lrsch_kwargs['eta_min'])
    elif lrsch_name == 'cosine_annealing_warmup':
        # lrsch = CosineWarmupLR
        lrsch = get_cosine_schedule_with_warmup
        # lrsch_kwargs['iter_in_one_epoch'] = train_loader_len
    elif lrsch_name == 'cos':
        lrsch = CosineAnnealingWarmRestarts
    elif lrsch_name == 'one_cycle':
        lrsch = torch.optim.lr_scheduler.OneCycleLR
        lrsch_kwargs['steps_per_epoch'] = train_loader_len
    elif lrsch_name == 'exp':
        lrsch = torch.optim.lr_scheduler.ExponentialLR
    else:
        raise Exception(f"Scheduler {lrsch_name} is not supported")
    return lrsch(optimizer, **lrsch_kwargs)

def get_cosine_schedule_with_warmup(optimizer, total_epochs, base_lr, min_lr=1e-6, warmup_epochs=20):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1. + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda)

# def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
#                      start_warmup_value=0, warmup_steps=-1):
        
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs * niter_per_ep
#     if warmup_steps > 0:
#         warmup_iters = warmup_steps
#     print("Set warmup steps = %d" % warmup_iters)
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

#     iters = np.arange(epochs * niter_per_ep - warmup_iters)
#     schedule = np.array(
#         [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

#     schedule = np.concatenate((warmup_schedule, schedule))

#     assert len(schedule) == epochs * niter_per_ep
#     return schedule
        
    
