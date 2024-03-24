import torch 
from torch import nn 
from torchvision.models.vgg import vgg16, VGG16_Weights

class GeneratorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
            
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
    
    def forward(self, out_labels, out_images, target_images):
        """ Calculate perception loss
        
        Args:
            out_labels: 判别器的输出
            out_images: 生成图像
            target_images: GT
        """
        # Adversarial loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        return 0.001 * adversarial_loss + 0.006 * perception_loss


    