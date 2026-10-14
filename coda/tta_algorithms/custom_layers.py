import torch
import torch.nn as nn

def replace_batchnorm_with_custom(model, freq, device='cpu'):
    """
    Replace the existing BatchNorm2d layer with the custom _BatchNorm2.
    And init weight, bias, running_mean, running_var with the existing BatchNorm2d layer.
    Refer to Section 3.3 TTA with FABN in the paper.
    - model: parent module
    - track_running_stats_hfc: If True TTA_v1.0.1, else TTA_v1.0.2
    - freq: frequency threshold for mask in adapt_bn
    """
    
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            custom_bn = _BatchNorm2d(
                num_features=module.num_features,
                affine=module.affine,
                training=module.training,
                momentum=module.momentum,
                eps=module.eps,
                freq=freq,
                device=device
                )
            setattr(model, name, custom_bn)
            if module.affine:
                custom_bn.weight.data = module.weight.data
                custom_bn.bias.data = module.bias.data
            custom_bn.running_mean.data = module.running_mean.data
            custom_bn.running_var.data = module.running_var.data
        else:
            replace_batchnorm_with_custom(module, freq, device)



def fft3channels_batch_parallel2(batch, batch_size, input_channels, mask, device): # 4d tensor, 2d tensor which is low-pass filter
    if mask is None:
        return batch

    # set mask 
    mask_low = _expand_mask_dim(mask, batch_size, input_channels).to(device)
    mask_high = _expand_mask_dim(1.0 - mask, batch_size, input_channels).to(device)
    
    fd = torch.fft.fftshift(torch.fft.fftn(batch, dim=(-2, -1)), dim=(-2, -1))

    # apply low-pass
    fd_low = fd * mask_low
    fd_low = torch.fft.ifftn(torch.fft.ifftshift(fd_low, dim=(-2, -1)), dim=(-2, -1)).real
    
    # apply high-pass
    fd_high = fd * mask_high
    fd_high = torch.fft.ifftn(torch.fft.ifftshift(fd_high, dim=(-2, -1)), dim=(-2, -1)).real
    
    return fd_low, fd_high     

def set_mask(input_size, low_r, high_r, device):
    # create x, y coordinates
    x = torch.arange(input_size, device=device).view(-1, 1).repeat(1, input_size)
    y = torch.arange(input_size, device=device).view(1, -1).repeat(input_size, 1)
    
    # calculate distance from the center
    center = input_size // 2
    distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # set mask based on the distance
    mask = (distance >= low_r) & (distance < high_r)
    mask= mask.float()

    return mask     
def _expand_mask_dim(mask_2d, batch_size, input_channels):
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)로 만듦
    mask_4d = mask_4d.repeat(batch_size, input_channels, 1, 1)  # (B, C, H, W)로 확장
    
    return mask_4d

def _ema(x, running_stat, momentum):   # x_new = (1-momentum) * x_{t-1} + momentum * x_t
    # running_stat = running_stat.to('cuda:0')
    return (1 - momentum) * running_stat + momentum * x  

class _BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, 
                num_features,
                affine,
                training, 
                momentum,
                eps,
                freq,
                device):
        """
        - num_features: number of channels
        - affine: to have affine weight and bias (y = W * x + b) # if none, y = x
        - training: BN training mode or not in 'torch.batch_norm'
                    * training mode: use batch stats in normalization 
                    * eval mode: use running stats
        - momentum: for EMA
        - eps: for EMA
        - freq: frequency threshold for mask
        """
        super(_BatchNorm2d, self).__init__(num_features)
        
        self.training = training # used in torch.batch_norm
        self.register_buffer("running_mean", torch.zeros(num_features))     
        self.register_buffer("running_var", torch.ones(num_features))
        
        self.affine = affine   # to have affine weight and bias (y = W * x + b) # if none, y = x
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            self.bias = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.momentum = momentum
        self.eps = eps
        self.freq = freq    # used in adapt_bn

        self.running_mean_lfc = None 
        self.running_var_lfc = None 
        self.running_mean_hfc = None 
        self.running_var_hfc = None 
        
        self.track_running_stats_hfc = True
        self.device = device

    
    def ema_freq(self, sample_mean, sample_var, running_mean, running_var, 
                 track_running_stat = False):           
        if track_running_stat and (running_mean is not None) and (running_var is not None):
            return _ema(sample_mean, running_mean, self.momentum), _ema(sample_var, running_var, self.momentum)
        else:
            return sample_mean, sample_var

    def adapt_bn(self, x):
        batch_size, input_channels, input_size, _ = x.shape
        assert input_channels == self.num_features

        if input_size == 112:
            # Step 1. x -> x_lfc, x_hfc            
            mask_base = set_mask(input_size, 0, self.freq, self.device).to(self.device) # input size, lowr, highr
            x_lfc, x_hfc = fft3channels_batch_parallel2(x, batch_size, input_channels, mask_base, self.device)
            
            # Step 2 & 3. x_lfc -> ema_freq(x_lfc,...), x_hfc -> ema_freq(x_hfc, ...)            
            batch_mean_lfc = x_lfc.mean(dim=(0,2,3), keepdim=False) # (channels,)
            batch_var_lfc = x_lfc.var(dim=(0,2,3), keepdim=False)
            batch_mean_hfc = x_hfc.mean(dim=(0,2,3), keepdim=False)
            batch_var_hfc = x_hfc.var(dim=(0,2,3), keepdim=False)
            
            with torch.no_grad():  # do not update running stats using loss
                self.running_mean_lfc, self.running_var_lfc = self.ema_freq(batch_mean_lfc, batch_var_lfc, 
                                                                            self.running_mean_lfc, self.running_var_lfc, # start from running_mean_init, running_var_init
                                                                            track_running_stat = True)  # use per batch stats
                self.running_mean_hfc, self.running_var_hfc = self.ema_freq(batch_mean_hfc, batch_var_hfc, 
                                                                            self.running_mean_hfc, self.running_var_hfc, # start from None, None
                                                                            track_running_stat = self.track_running_stats_hfc)  # use per batch stats
                # Step 4. Combine stats from lfc and hfc
                self.running_mean = self.running_mean_lfc + self.running_mean_hfc
                self.running_var = self.running_var_lfc + self.running_var_hfc - 2 * self.running_mean_lfc * self.running_mean_hfc  


        else:
            with torch.no_grad():  # do not update running stats using loss
                batch_mean = x.mean(dim=(0, 2, 3))
                batch_var = x.var(dim=(0, 2, 3))
                self.running_mean, self.running_var = self.ema_freq(batch_mean, batch_var, 
                                                                self.running_mean, self.running_var, # start from running_mean_init, running_var_init
                                                                track_running_stat = True)  # use per batch stats


    def forward(self,x):
        self.adapt_bn(x) 
        
        return torch.batch_norm(   # source code in https://github.com/pytorch/pytorch/blob/420b37f3c67950ed93cd8aa7a12e673fcfc5567b/aten/src/ATen/native/Normalization.cpp#L61-L126
                x,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.training, # Use running stats (self.training = False) or batch stats (self.training = True) in normalization
                self.momentum,
                self.eps,
                torch.backends.cudnn.enabled,
        )     
