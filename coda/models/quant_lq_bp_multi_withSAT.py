import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

"""
Straight-Through Estimator (STE) in Quantization-Aware Training (QAT)
"""
# STE for [BMVC '20] Scale-Adjusted Training (SAT)
# Refer to https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L11
class SATSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x, ei):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        y = torch.round(ei * x)
        y.div_(ei)
        assert torch.all(y >= 0) and torch.all(y <= 1)
        return y

    @staticmethod
    def backward(ctx, dy):  # dy stands for dL/dy
        # dx = dy (estimating dy/dx as 1)
        # dei: not defined (None)
        return dy, None     # Return dx, dei

# STE 1 for [ICLR '20] Learned Step size Quantization (LSQ)
# Refer to https://github.com/hustzxd/LSQuantization/blob/master/lsq.py#L47
def identity_with_grad_scaling(x, scale):
    y_fp = x            # At forward prop
    y_bp = x * scale    # At back-prop 
    return y_fp.detach() - y_bp.detach() + y_bp  # detach(): requires_grad=False

# STE 2 for LSQ
# Refer to https://github.com/hustzxd/LSQuantization/blob/master/lsq.py#L53
def round_with_grad_pass(x):
    y_fp = x.round()    # At forward prop
    y_bp = x            # At back-prop
    return y_fp.detach() - y_bp.detach() + y_bp

"""
Fake-quantizer in Quantization-Aware Training (QAT)
"""
# Refer to https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L137 and https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L238
class SATWeightFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        out_features=None):
        super(SATWeightFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        self.ei = 2 ** bits - 1
        self.out_features = out_features
        self.weight_scale = -1.0    # Dummy

    def forward(self, w):
        w_fq = torch.tanh(w) / torch.max(torch.abs(torch.tanh(w)))
        w_fq.add_(1.0).div_(2.0)
        w_fq = SATSTE.apply(w_fq, self.ei)
        w_fq.mul_(2.0).sub_(1.0)
        if self.out_features is not None: # used in QLinear
            self.weight_scale = 1.0 / (self.out_features) ** 0.5
            self.weight_scale = self.weight_scale / torch.std(w_fq.detach())
            if self.training: w_fq.mul_(self.weight_scale)
        return w_fq

# Refer to https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L171 and https://github.com/deJQK/AdaBits/blob/master/models/quant_ops.py#L276
class SATActFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        init_alpha):
        super(SATActFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        self.ei = 2 ** bits - 1
        self.alpha = nn.Parameter(torch.tensor(init_alpha))     # Initial value of alpha: 8.0 for conv layer, 10.0 for FC layer

    def forward(self, x):
        alpha = torch.abs(self.alpha)
        x_fq = torch.relu(x)
        x_fq = torch.where(x_fq < alpha, x_fq, alpha) # PACT
        x_fq.div_(alpha)
        x_fq = SATSTE.apply(x_fq, self.ei)
        x_fq.mul_(alpha)
        return x_fq

# Refer to https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L61 and https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L107
# per_channel is False


class LSQWeightFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        all_positive=False,
        symmetric=False):
        super(LSQWeightFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.n = 0
            self.p = 2 ** bits - 1
        else:
            if symmetric:
                self.n = - 2 ** (bits - 1) + 1
                self.p = 2 ** (bits - 1) - 1
            else:
                self.n = - 2 ** (bits - 1)
                self.p = 2 ** (bits - 1) - 1

        self.s = nn.Parameter(torch.tensor(-1.0))   # Dummy

    def forward(self, w):
        grad_scale = 1.0 / math.sqrt(w.numel() * self.p)

        if self.s == -1.0:
            self.s.data = 2.0 * w.abs().mean() / math.sqrt(self.p) # initialization

        s = identity_with_grad_scaling(self.s, grad_scale) # step size
        # w_q = round_with_grad_pass((w / s).clamp(self.n, self.p))
        w_fq = round_with_grad_pass((w / s).clamp(self.n, self.p)) * s
        return w_fq

class LSQWeightFakeQuantizer_perchannel(nn.Module):
    def __init__(self, bits, all_positive=False, symmetric=True):
        super(LSQWeightFakeQuantizer_perchannel, self).__init__()

        assert bits in [i for i in range(2, 32)]
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.n = 0
            self.p = 2 ** bits - 1
        else:
            if symmetric:
                self.n = -2 ** (bits - 1) + 1
                self.p = 2 ** (bits - 1) - 1
            else:
                self.n = -2 ** (bits - 1)
                self.p = 2 ** (bits - 1) - 1

        self.bits = bits
        self.symmetric = symmetric
        self.all_positive = all_positive
        self.register_parameter("s", None)  # initialize later

    def forward(self, w):
        # assume Conv2d or Linear: weight shape (out_channels, ...)
        assert w.dim() >= 2
        c = w.size(0)  # output channels

        # Lazy init s
        if self.s is None:
            s_init = 2.0 * w.abs().mean(dim=tuple(range(1, w.dim())), keepdim=False) / math.sqrt(self.p)
            self.s = nn.Parameter(s_init.detach())

        # Compute gradient scale per channel
        numel_per_channel = w[0].numel()
        grad_scale = 1.0 / math.sqrt(numel_per_channel * self.p)
        s_scaled = identity_with_grad_scaling(self.s, grad_scale)

        # reshape s for broadcasting
        view_shape = [c] + [1] * (w.dim() - 1)
        s_scaled = s_scaled.view(*view_shape)

        w_q = round_with_grad_pass((w / s_scaled).clamp(self.n, self.p)) * s_scaled
        return w_q



class LSQWeightFakeQuantizer_constant_rescale(nn.Module):
    def __init__(
        self,
        bits,
        all_positive=False,
        symmetric=False,
        out_features=None):
        super(LSQWeightFakeQuantizer_constant_rescale, self).__init__()

        assert bits in [i for i in range(2, 32)]
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            self.n = 0
            self.p = 2 ** bits - 1
        else:
            if symmetric:
                self.n = - 2 ** (bits - 1) + 1
                self.p = 2 ** (bits - 1) - 1
            else:
                self.n = - 2 ** (bits - 1)
                self.p = 2 ** (bits - 1) - 1

        self.s = nn.Parameter(torch.tensor(-1.0))   # Dummy
        self.out_features = out_features # only for comparison to SAT!!!

    def forward(self, w):
        grad_scale = 1.0 / math.sqrt(w.numel() * self.p)

        if self.s == -1.0:
            self.s.data = 2.0 * w.abs().mean() / math.sqrt(self.p) # initialization

        s = identity_with_grad_scaling(self.s, grad_scale) # step size
        # w_q = round_with_grad_pass((w / s).clamp(self.n, self.p))
        w_fq = round_with_grad_pass((w / s).clamp(self.n, self.p)) * s
        if self.out_features is not None:
            self.weight_scale = 1.0 / (self.out_features) ** 0.5
            self.weight_scale = self.weight_scale / torch.std(w_fq.detach())
            if self.training: w_fq.mul_(self.weight_scale)
        return w_fq

# Refer to https://github.com/hustzxd/EfficientPyTorch/blob/clean/models/_modules/lsq.py#L136
class LSQActFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        symmetric=False):
        super(LSQActFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        self.bits = bits
        self.symmetric = symmetric
        self.n = -1.0                               # Dummy
        self.p = -1.0                               # Dummy

        self.s = nn.Parameter(torch.tensor(-1.0))   # Dummy

    def forward(self, x):
        if self.p == -1.0:
            all_positive = x.min() >= -1e-5
            if all_positive:
                assert not self.symmetric, "Positive quantization cannot be symmetric"
                self.n = 0
                self.p = 2 ** self.bits - 1
            else:
                if self.symmetric:
                    self.n = - 2 ** (self.bits - 1) + 1
                    self.p = 2 ** (self.bits - 1) - 1
                else:
                    self.n = - 2 ** (self.bits - 1)
                    self.p = 2 ** (self.bits - 1) - 1

        grad_scale = 1.0 / math.sqrt(x.numel() * self.p)

        if self.s == -1.0:
            self.s.data = 2.0 * x.abs().mean() / math.sqrt(self.p)

        s = identity_with_grad_scaling(self.s, grad_scale)
        # x_q = round_with_grad_pass((x / s).clamp(self.n, self.p))
        x_fq = round_with_grad_pass((x / s).clamp(self.n, self.p)) * s
        return x_fq


class LSQActFakeQuantizer_pertensor(nn.Module):
    def __init__(self, bits):
        super(LSQActFakeQuantizer_pertensor, self).__init__()
        assert bits in [i for i in range(2, 32)]
        self.bits = bits
        self.qmin = 0
        self.qmax = 2 ** bits - 1

        self.register_parameter("s", None)  # step size
        self.register_parameter("z", None)  # zero point

    def forward(self, x):
        if self.s is None or self.z is None:
            # Initialize using min/max range (asymmetric)
            x_min = x.min().detach()
            x_max = x.max().detach()
            s_init = (x_max - x_min) / (self.qmax - self.qmin + 1e-8)
            z_init = (-x_min / s_init).round().clamp(self.qmin, self.qmax)

            self.s = nn.Parameter(s_init)
            self.z = nn.Parameter(z_init)

        grad_scale = 1.0 / math.sqrt(x.numel() * self.qmax)
        s = identity_with_grad_scaling(self.s, grad_scale)
        z = self.z  # no grad needed, fixed in most QAT

        # Quantize and dequantize (fake quant)
        x_int = round_with_grad_pass((x / s) + z)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_fq = (x_int - z) * s
        return x_fq



MOVING_AVERAGES_FACTOR = 0.9
EPS = 0.0001
NORM_PPF_0_75 = 0.6745

class LQWeightFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits,
        out_channels,
        kernel_size):
        super(LQWeightFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        self.bits = bits
        self.num_levels = 2**self.bits
        init_thrs_multiplier = []
        for i in range(1, self.num_levels):
            thrs_multiplier_i = [0. for j in range(self.num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        self.thrs_multiplier = nn.Parameter(torch.zeros(self.num_levels - 1, self.num_levels), requires_grad=False)
        self.thrs_multiplier.data = torch.FloatTensor(init_thrs_multiplier)
        self.out_channels = out_channels
        self.basis = nn.Parameter(torch.ones(self.bits, self.out_channels))   # Initial
        self.codec_vector = nn.Parameter(torch.ones(self.num_levels, self.bits), requires_grad=False)
        self.codec_index = nn.Parameter(torch.ones(self.num_levels, dtype=torch.int), requires_grad=False)
        init_basis = []
        self.n = kernel_size*kernel_size*out_channels
        base = NORM_PPF_0_75*((2./self.n)**0.5)/(2**(self.bits-1))
        for i in range(self.bits):
            init_basis.append([(2 ** i) * base for j in range(self.out_channels)])
        self.basis.data = torch.FloatTensor(init_basis)

        # initialize level_codes
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [0. for j in range(self.bits)]
            level_number = i
            for j in range(self.bits):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.codec_vector.data = torch.FloatTensor(init_level_multiplier)

        init_codec_index = []
        for i in range(self.num_levels):
            init_codec_index.append(i)
        self.codec_index.data= torch.IntTensor(init_codec_index)
        
    def forward(self, inputs):
        levels = torch.matmul(self.codec_vector, self.basis.data)
        levels, sort_id = torch.sort(levels, 0, descending=False)
        # calculate threshold
        thrs = torch.matmul(self.thrs_multiplier, levels)
        # calculate output y and its binary codec
        origin_shape = inputs.shape
        x = inputs.reshape(self.out_channels, -1)

        y = levels[0].unsqueeze(1).expand_as(x)  # output
        codec = self.codec_index[sort_id[0]].unsqueeze(1).expand_as(x)
        for i in range(self.num_levels - 1):
            g = x > thrs[i].unsqueeze(1).expand_as(x)
            y = torch.where(g, levels[i + 1].unsqueeze(1).expand_as(x), y)
            codec = torch.where(g, self.codec_index[sort_id[i+1]].unsqueeze(1).expand_as(codec), codec)

        y = y.reshape(origin_shape)
        if not self.training:
            return y

        y = inputs - inputs.detach() + y.detach()
        return y

class LQActFakeQuantizer(nn.Module):
    def __init__(
        self,
        bits):
        super(LQActFakeQuantizer, self).__init__()

        assert bits in [i for i in range(2, 32)]
        self.bits = bits
        self.num_levels = 2**self.bits
        init_thrs_multiplier = []
        for i in range(1, self.num_levels):
            thrs_multiplier_i = [0. for j in range(self.num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        self.thrs_multiplier = nn.Parameter(torch.zeros(self.num_levels - 1, self.num_levels), requires_grad=False)
        self.thrs_multiplier.data = torch.FloatTensor(init_thrs_multiplier)
        self.basis = nn.Parameter(torch.ones(self.bits, 1))   # Initial
        self.codec_vector = nn.Parameter(torch.ones(self.num_levels, self.bits), requires_grad=False)
        self.codec_index = nn.Parameter(torch.ones(self.num_levels, dtype=torch.int), requires_grad=False)
        init_basis = []
        base = NORM_PPF_0_75 * 2. / (2 ** (self.bits - 1))
        for i in range(self.bits):
            init_basis.append([(2 ** i) * base for j in range(1)])
        self.basis.data = torch.FloatTensor(init_basis)

        # initialize level_codes
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [0. for j in range(self.bits)]
            level_number = i
            for j in range(self.bits):
                binary_code = level_number % 2
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.codec_vector.data = torch.FloatTensor(init_level_multiplier)

        init_codec_index = []
        for i in range(self.num_levels):
            init_codec_index.append(i)
        self.codec_index.data= torch.IntTensor(init_codec_index)


    def forward(self, inputs):
        # calculate levels and sort
        levels = torch.matmul(self.codec_vector, self.basis.data)
        levels, sort_id = torch.sort(levels, 0, descending=False)

        # calculate threshold
        thrs = torch.matmul(self.thrs_multiplier, levels)

        x = inputs
        # calculate output y and its binary code
        x_shape = x.shape
        x = x.reshape(1, -1)
        y = levels[0].unsqueeze(1).expand_as(x)
        codec = self.codec_index[sort_id[0]].unsqueeze(1).expand_as(x)
        for i in range(self.num_levels - 1):
            g = x > thrs[i].unsqueeze(1).expand_as(x)
            y = torch.where(g, levels[i+1].unsqueeze(1).expand_as(x), y)
            codec = torch.where(g, self.codec_index[sort_id[i+1]].unsqueeze(1).expand_as(codec), codec)
        y = y.reshape(x_shape)
        if not self.training:
            return y

        x_clip = torch.minimum(inputs, levels[self.num_levels - 1])  # gradient clip
        y = x_clip - x_clip.detach() + y.detach()  # gradient: y=clip(x)

        return y

"""
Etc.
"""
def get_extra_q_bits(scheme, bits_w, bits_a):
    if scheme == 'sat':
        bits_w_first = max(bits_w, 8) if bits_w is not None else None
        bits_a_first = None     # Practically 8 if bits_a is not None, since model input will be in uint8 after multiplied by 255 then rounded (as data augmentation)
        bits_w_last = max(bits_w, 8) if bits_w is not None else None
        bits_a_last = bits_a
    elif scheme == 'lsq':
        bits_w_first = max(bits_w, 8) if bits_w is not None else None
        bits_a_first = max(bits_a, 8) if bits_a is not None else None
        bits_w_last = max(bits_w, 8) if bits_w is not None else None
        bits_a_last = max(bits_a, 8) if bits_a is not None else None
    elif scheme == 'lq':
        bits_w_first = None
        bits_a_first = None
        bits_w_last = None
        bits_a_last = None
    else:
        assert scheme is None
        bits_w_first = None
        bits_a_first = None
        bits_w_last = None
        bits_a_last = None

    return bits_w_first, bits_a_first, bits_w_last, bits_a_last