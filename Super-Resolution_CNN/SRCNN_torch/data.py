import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

def is_image_file(filename):
    '''
    用来检查文件名filename是否以.png、.jpg或.jpeg中的任何一个扩展名结尾
    '''
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])

def load_img(filepath):
    '''
    从给定的文件路径 filepath 加载图像，并将图像转换为 YCbCr 色彩空间
    返回Y通道，即图像的亮度信息
    '''
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

CROP_SIZE = 32

class DatasetFromFolder(Dataset):
    """从指定的图像目录中加载图像，并对这些图像应用一系列的转换"""
    def __init__(self, image_dir, zoom_factor) -> None:
        """
        image_dir:图像的目录路径
        zoom_factor:一个用于调整图像缩放的因子
        """
        super().__init__()
        
        ### 通过listdir获取图像目录中的所有文件，并通过is_image_file函数过滤出图像文件
        ### image_filenames将包含所有图像文件的路径
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        
        ### 计算有效的裁剪大小，确保它是zoom_factor的倍数
        crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor)  # cropping the image
        
        ### 定义输入图像的转换，包括中心裁剪、下采样、上采样（使用双三次插值）和转换成张量
        self.input_transform = transforms.Compose(
                                [transforms.CenterCrop(crop_size), # cropping the image 中心裁剪
                                transforms.Resize(crop_size//zoom_factor), # subsampling the image(half size) 下采样
                                transforms.Resize(crop_size, interpolation=Image.BICUBIC), 
                                # bicubic upsampling to get back the original size 上采样
                                transforms.ToTensor() # 转换成张量
                                ])
        
        ### 定义目标图像的转换，包括中心裁剪和转换成张量，这里的目的是保持目标图像的原始质量，不进行缩放
        self.target_transform = transforms.Compose(
                                [transforms.CenterCrop(crop_size), # since it's the target, we keep its priginal quality
                                 transforms.ToTensor()
                                ])
        ### 上述操作保证输入图像和输出图像大小一致,且输入图像为低分辨率图像
        
    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        
        # 高斯核对子图像进行模糊处理
        # input = input.filter(ImageFilter.GaussianBlur(1))
        input = self.input_transform(input)
        target = self.target_transform(target)
        
        return input, target
    
    def __len__(self):
        return len(self.image_filenames)
        
if __name__ == '__main__':
    
    folder = os.path.dirname(__file__)
    path = os.path.join(folder, './data/train')
    print(path)
    data = DatasetFromFolder(path, 2)
    
    train_dataloader = DataLoader(dataset=data, batch_size=4, shuffle=True)
    
    print(f'data_size: {len(train_dataloader.dataset)}')
    print(f'num_batch: {len(train_dataloader)}')
    X, y = next(iter(train_dataloader))
    print(f'input shape: {X.shape}')
    print(f'out shape: {y.shape}')
    
    
    
    
    
        
        

