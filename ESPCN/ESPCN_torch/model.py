import torch
import torch.nn as nn
import torch.nn.functional as F 

class ESPCNModel(nn.Module):
    def __init__(self, upscale_factor) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # n x 32 x h x w ===> n x 9 x h x w
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), kernel_size=(3, 3), 
                               stride=(1, 1), padding=(1, 1))
        # n x 9 x h x w ===> n x 1 x 3h x 3w
        self.pixel_shuffle= nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        
        return x
    
if __name__== "__main__":
    model = ESPCNModel(upscale_factor=3)
    print(model)
    
    pixel_shuffle = nn.PixelShuffle(3)
    input = torch.randn(2, 18, 4, 4) # n c h w
    print(f'input: {input}')
    # c_in=9   h_in=4   w_in=4
    # c_out = c_in / upscale^2
    # h_out = h_in * upscale    w_out = w_in * upscale
    output = pixel_shuffle(input)
    print(output.size()) # torch.Size([1, 1, 12, 12])
    print(f'output: {output}')