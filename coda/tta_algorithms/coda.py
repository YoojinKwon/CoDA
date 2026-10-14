import torch.nn as nn

from tta_algorithms.custom_layers_coda import replace_batchnorm_with_custom


def configure_model_for_fabn(model, freq, device='cuda'):   
    assert freq > 0, "tta requires frequency threshold > 0"
    replace_batchnorm_with_custom(model, freq, 0, device=device) 
    model.eval() # use EMA result in torch.batch_norm
    model.requires_grad_(False) # do not update weight or bn affine params

    return model

class FABN(nn.Module):
    def __init__(self, model):
        super().__init__()
        
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)           
        
        return outputs