import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        
        return out

