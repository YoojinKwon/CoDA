import torch
import torch.nn as nn
from tta_algorithms.custom_layers import replace_batchnorm_with_custom


def configure_model(model, freq, bn_train, track_running_stats_hfc, device=None, avg_step=0):
    """
    train: If True, TTA_v1.1 else TTA_v1.0
    track_running_stats_hfc: If True, TTA_v1.0.1 and TTA_v1.0.2 else TTA_v1.0.0
    avg_step: If avg_step > 0, TTA_v1.0.1
    """
    
    assert freq > 0, "tta requires frequency threshold > 0"
        
    replace_batchnorm_with_custom(model, freq, track_running_stats_hfc, device=device, avg_step=avg_step)

    if bn_train:
        model.train()
    else:
        model.eval() # use EMA result in torch.batch_norm

    model.requires_grad_(False) # do not update weight or bn affine params
    
    for m in model.modules(): # Configure norm for Tent updates: enable grad + force batch statisics (refer to tent)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)

    return model

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

# Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L43
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class TENT_FABN(nn.Module):
    def __init__(
        self, 
        model,
        optimizer,  
        adapt_step=1  
    ):
        super().__init__()
        
        self.model = model
        self.optim = optimizer
        self.adapt_step = adapt_step
        assert self.adapt_step > 0, "requires >= 1 step(s) to forward and update"
        
    def forward(self, x):
        for _ in range(self.adapt_step):
            outputs = forward_and_adapt(x, self.model, self.optim)    
        
        return outputs
