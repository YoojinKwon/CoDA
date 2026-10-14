"""
Implementation based on the paper 'Improving robustness against common corruptions by covariate shift adaptation', NeurIPS 2020
# BN1d added
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Norm_BN(nn.Module):
    def __init__(
        self,
        num_features, 
        eps, 
        alpha, 
        affine
        ):
        super(Norm_BN, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.alpha = alpha
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))     # \hat{mu}_{source}
        self.register_buffer("running_var", torch.ones(num_features))       # \hat{sigma}^2_{source}

    def forward(self, x):
        assert not self.training   # Use running stats in normalization
        
        if x.dim() == 3:
            x = x.permute(0, 2, 1)  # 'batch_size tokens channels -> batch_size channels tokens'
            
            batch_mean = torch.mean(x, dim=[0, 2])   # mu_{target}
            batch_var = torch.var(x, dim=[0, 2])     # sigma^2_{target}

            x = F.batch_norm(
                x,
                (1.0 - self.alpha)*self.running_mean + self.alpha*batch_mean,
                (1.0 - self.alpha)*self.running_var + self.alpha*batch_var,
                weight=self.weight,
                bias=self.bias,
                training=False,   # Use running stats in normalization
                momentum=0,
                eps=self.eps)

            x = x.permute(0, 2, 1)  # 'batch_size channels tokens -> batch_size tokens channels'
            return x

        else:
            batch_mean = torch.mean(x, dim=[0, 2, 3])   # mu_{target}
            batch_var = torch.var(x, dim=[0, 2, 3])     # sigma^2_{target}

            x = F.batch_norm(
                x,
                (1.0 - self.alpha)*self.running_mean + self.alpha*batch_mean,
                (1.0 - self.alpha)*self.running_var + self.alpha*batch_var,
                weight=self.weight,
                bias=self.bias,
                training=False,   # Use running stats in normalization
                momentum=0,
                eps=self.eps)

            return x


def replace_all_bn(module, alpha=0.1):  # alpha = n / (N + n)
    for child_name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new_bn = Norm_BN(
                child.num_features, 
                child.eps, 
                alpha, 
                child.affine
            )
            if child.affine:
                new_bn.weight.data = child.weight.data
                new_bn.bias.data = child.bias.data
            new_bn.running_mean.data = child.running_mean
            new_bn.running_var.data = child.running_var
            setattr(module, child_name, new_bn)
        
        if isinstance(child, nn.BatchNorm1d):
            new_bn = Norm_BN(
                child.num_features, 
                child.eps, 
                alpha, 
                child.affine
            )
            if child.affine:
                new_bn.weight.data = child.weight.data
                new_bn.bias.data = child.bias.data
            new_bn.running_mean.data = child.running_mean
            new_bn.running_var.data = child.running_var
            setattr(module, child_name, new_bn)
        else:
            replace_all_bn(child)

 
class Norm(nn.Module):
    def __init__(self, 
                 model, 
                 alpha=0.1,
                 reset_stats=False):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, x):
        replace_all_bn(self.model, self.alpha)
        self.model.eval()
        outputs = self.model(x)
        
        return outputs     
   