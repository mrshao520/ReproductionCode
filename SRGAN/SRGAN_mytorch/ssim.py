from math import exp
import torch
import torch.nn.functional as F 
from torch.autograd import Variable

def gaussian(window_size, sigma):
    """
    计算一维的高斯分布向量
    
    Args:
        window_size: 窗口大小
        sigma: 标准差
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    通过两个一维高斯分布向量进行矩阵乘法创建高斯核，可以设定channel参数扩展为3通道
    
    Args:
        window_size: 窗口大小
        channel: 通道数
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # print(f'_1D_window: {_1D_window}')
    # 将一维的高斯窗与它的转置进行矩阵乘法（mm 方法）
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # print(f'_2D_window: {_2D_window}')
    # 在第一个维度（通道维度）上扩展 channel 次，在第二个维度上扩展1次，在第三个和第四个维度上保持窗口大小不变。
    # 使用 contiguous() 方法确保张量内存是连续的
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # print(f'window: {window}')
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)



if __name__ == '__main__':
    print(gaussian(5, 1).unsqueeze(1))
    create_window(5, 2)