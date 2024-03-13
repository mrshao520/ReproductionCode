import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor, feature_dimension, shrinking, mapping_layers) -> None:
        """
        feature_dimension: LR特征维度的数量
        shrinking: 收缩程度
        mapping_layers: 映射层数
        """
        super().__init__()
        
        # feature_extractions
        self.model_seq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=feature_dimension,
                        kernel_size=5, padding=2),
            nn.PReLU(feature_dimension))
        
        # shrinking
        self.model_seq.extend([
            nn.Conv2d(in_channels=feature_dimension,
                        out_channels=shrinking, kernel_size=1),
            nn.PReLU(shrinking)
        ])
        
        # non-linear mapping
        for i in range(mapping_layers):
            self.model_seq.extend([
                nn.Conv2d(in_channels=shrinking, out_channels=shrinking,
                            kernel_size=3, padding=1),
                nn.PReLU(shrinking)
            ])
        
        # expending
        self.model_seq.extend([
            nn.Conv2d(in_channels=shrinking, out_channels=feature_dimension, 
                        kernel_size=1),
            nn.PReLU(feature_dimension)
        ])
        
        # deconvolution
        self.last_seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_dimension, out_channels=1, kernel_size=9, 
                                stride=upscale_factor, padding=4,
                                output_padding=upscale_factor-1))
                 
    
        
    def forward(self, x):
        x = self.model_seq(x)
        x = self.last_seq(x)
        
        return x

if __name__ == '__main__':
    model = FSRCNN(3, 56, 12, 4)
    print(f'model ---------\n{model}')
    
    x = torch.randn((1, 1, 7, 7))
    x = model(x)
    print(f'x shape : {x.shape}')
    print(f'x : {x}')
