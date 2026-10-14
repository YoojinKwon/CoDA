"""
Implementation of baseline tta
Norm v1
Norm v2
Tent
# BN1d added
"""

from copy import deepcopy

import torch
import torch.nn as nn


"""
Tent
"""

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L83
def copy_model_and_optimizer(model, optimizer):
    model_state_dict = deepcopy(model.state_dict())
    optimizer_state_dict = deepcopy(optimizer.state_dict())
    return model_state_dict, optimizer_state_dict

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L43
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L49
@torch.enable_grad()  # NOTE: ensure grads in possible no_grad context for testing
def forward_and_adapt(x, model, optimizer):
    # Forward
    outputs = model(x)
    # Adaptation
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L90
def load_model_and_optimizer(model, optimizer, model_state_dict, optimizer_state_dict):
    model.load_state_dict(model_state_dict, strict=True)
    optimizer.load_state_dict(optimizer_state_dict)

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L8
class Tent(nn.Module):
    def __init__(
        self, 
        model, 
        optimizer, 
        steps=1, 
        episodic=False):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.cur_steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state_dict, self.optimizer_state_dict = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.cur_steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if (self.model_state_dict is None) or (self.optimizer_state_dict is None):
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer,
            self.model_state_dict, self.optimizer_state_dict)

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L96
def configure_model_for_tent(model):
    # Set to training mode, as Tent optimizes the model to minimize entropy
    model.train() # Use batch stat in normalization
    # Disable grad, to (re-)enable only what Tent updates
    model.requires_grad_(False)
    # Configure norm for Tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            # Force use of batch stats in train and eval modes
            m.track_running_stats = False # do not use EMA
            m.running_mean = None
            m.running_var = None

    return model

def configure_optimizer_for_tent(dataset):
    # Set to training mode, as Tent optimizes the model to minimize entropy
    if dataset in ['cifar10', 'cifar10c']:
        optimizer = torch.optim.Adam
        kwargs_optim = {'lr': 0.001}
        
    elif dataset in ['imagenet', 'imagenetc', 'imagenet_es', 'imagenetr', 'tiny_imagenet', 'tiny_imagenetc']:
        optimizer = torch.optim.SGD
        kwargs_optim = {'lr': 0.00025}
    else:
        raise Exception(f"Optimizer type is not supported.")
    return optimizer, kwargs_optim


# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L64
def collect_params(model):
    params = []
    param_names = []
    for m_name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for param_name, param in m.named_parameters():
                if param_name in ['weight', 'bias']:  # Weight is scale; Bias is shift
                    params.append(param)
                    param_names.append(f"{m_name}.{param_name}")
    return params, param_names



