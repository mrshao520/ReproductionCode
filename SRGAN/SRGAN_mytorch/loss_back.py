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
        self.tv_loss = TVLoss()
    
    def forward(self, out_labels, out_images, target_images):
        # Adversarial loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        iamge_loss = self.mse_loss(out_images, target_images)
        # TV loss
        tv_loss = self.tv_loss(out_images)
        
        return iamge_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1) -> None:
        super().__init__()
        
        self.tv_loss_weight = tv_loss_weight
        
    def forward(self, x):
        batch_size, _, h_x, w_x = tuple(x.size())
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
        
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
if __name__ == '__main__':
    example = torch.randint(0, 10, (2, 3, 4, 5))
    print(example)
    batch_size, _, h_x, w_x = tuple(example.size())
    count_h = TVLoss.tensor_size(example[:, :, 1:, :])
    count_w = TVLoss.tensor_size(example[:, :, :, 1:])
    print(f'count_h : {count_h} shape : {example[:, :, 1:, :].shape}')
    print(f'count_w : {count_w} shape : {example[:, :, :, 1:].shape}')
    h_tv = torch.pow((example[:, :, 1:, :] - example[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((example[:, :, :, 1:] - example[:, :, :, :w_x - 1]), 2).sum()
    print(f'h_tv : {h_tv} shape : {h_tv.shape}')
    print(f'w_tv : {w_tv} shape : {w_tv.shape}')
    reslut = 1 * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    print(f'reslut : {reslut} shape : {reslut.shape}')
    