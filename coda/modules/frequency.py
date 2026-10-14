import torch

### FFT (Refer to https://github.com/DequanWang/tent/blob/master/tent.py#L83)
def fft3channels_batch(batch, batch_size, input_channels, mask, args): # 4d tensor, 2d tensor which is low-pass filter
    if mask is None:
        return batch

    # set mask 
    mask_low = _expand_mask_dim(mask, batch_size, input_channels).cuda()
    mask_high = _expand_mask_dim(1.0 - mask, batch_size, input_channels).cuda()
    
    fd = torch.fft.fftshift(torch.fft.fftn(batch, dim=(-2, -1)), dim=(-2, -1))

    # apply low-pass
    fd_low = fd * mask_low
    fd_low = torch.fft.ifftn(torch.fft.ifftshift(fd_low, dim=(-2, -1)), dim=(-2, -1)).real
    
    # apply high-pass
    fd_high = fd * mask_high
    fd_high = torch.fft.ifftn(torch.fft.ifftshift(fd_high, dim=(-2, -1)), dim=(-2, -1)).real
    
    return fd_low, fd_high     

def set_mask(input_size, low_r, high_r):
    # x와 y 좌표 생성
    # x = torch.arange(input_size).view(-1, 1).repeat(1, input_size)
    # y = torch.arange(input_size).view(1, -1).repeat(input_size, 1)
    x = torch.arange(input_size, device='cuda').view(-1, 1).repeat(1, input_size)
    y = torch.arange(input_size, device='cuda').view(1, -1).repeat(input_size, 1)
    
    # 중심으로부터 거리 계산
    center = input_size // 2
    distance = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # 거리 조건에 따라 마스크 설정
    mask = (distance >= low_r) & (distance < high_r)
    mask= mask.float()

    return mask     
    
def _expand_mask_dim(mask_2d, batch_size, input_channels):
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)로 만듦
    mask_4d = mask_4d.repeat(batch_size, input_channels, 1, 1)  # (B, C, H, W)로 확장
    
    return mask_4d