import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset
import torchvision.transforms as transform
from PIL import Image, ImageOps
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) 
                for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG', '.bmp'])

def is_video_file(filename):
    return any(filename.endswith(extension) 
               for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def generate_dataset(in_dir, out_dir, upscale_factor, size_input, stride):
    """
    in_dir:         原始图像文件夹
    out_dir:        处理后的图像文件夹
    upscale_factor: 缩放比例
    size_input:     数据集中输入图像的大小
    stride:         步长
    """
    out_dir = out_dir + '/X' + str(upscale_factor)
    if os.path.exists(out_dir):
        return
    
    images_name = [x for x in listdir(in_dir) if is_image_file(x)]
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    input_path = out_dir + '/LR'
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    target_path = out_dir + '/HR'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    count = 1
    for image_name in images_name:
        input = Image.open(in_dir + '/' + image_name)
        target = input.copy()
        
        input = ImageOps.scale(input, 1/upscale_factor, 
                               resample=Image.Resampling.BICUBIC)
        in_width = input.width
        in_height = input.height
        target_size = size_input * upscale_factor
        for x in range(0, in_width - size_input + 1, stride):
                for y in range(0, in_height - size_input + 1, stride):
                    sub_input = input.crop((x, y, x + size_input, 
                                                 y + size_input))
                    
                    tar_x = x * upscale_factor
                    tar_y = y * upscale_factor
                    sub_target = target.crop((tar_x, tar_y, tar_x + target_size,
                                                 tar_y + target_size))
                    
                    sub_input.save(f'{out_dir}/LR/{count}_{image_name}')
                    sub_target.save(f'{out_dir}/HR/{count}_{image_name}')
                    count = count + 1
                    
        print(f'the number of images: {count}')
        
        
class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, 
                 input_transform=None, target_transform=None) -> None:
        super().__init__()
        # self.input_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data'
        # self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + 'target'
        self.input_dir = dataset_dir + '/LR'
        self.target_dir = dataset_dir + '/HR'
        # 获取图片
        self.input_filenames = [join(self.input_dir, x) 
                                for x in listdir(self.input_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) 
                                 for x in listdir(self.target_dir) if is_image_file(x)]
        # 图像预处理函数
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        input, _, _ = Image.open(self.input_filenames[index]).convert('YCbCr').split()
        target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            
        return input, target
    
    def __len__(self):
        return len(self.input_filenames)
        
class RGBDataset(Dataset):
    def __init__(self, dataset_dir, upscale_factor, 
                 input_transform=None, target_transform=None) -> None:
        super().__init__()
        # self.input_dir = dataset_dir + '/SRF_' + str(upscale_factor) + '/data'
        # self.target_dir = dataset_dir + '/SRF_' + str(upscale_factor) + 'target'
        self.input_dir = dataset_dir + '/LR'
        self.target_dir = dataset_dir + '/HR'
        self.upscale_factor = upscale_factor
        # 获取图片
        self.input_filenames = [join(self.input_dir, x) 
                                for x in listdir(self.input_dir) if is_image_file(x)]
        self.target_filenames = [join(self.target_dir, x) 
                                 for x in listdir(self.target_dir) if is_image_file(x)]
        # 图像预处理函数
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        
        input, in_cb, in_cr = Image.open(self.input_filenames[index]).convert('YCbCr').split()
        target, ta_cb, ta_cr = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            
        return input, target, in_cb, in_cr, ta_cb, ta_cr
    
    def __len__(self):
        return len(self.input_filenames)

if __name__ == '__main__':
    folder = os.path.dirname(__file__)
    data_dir = os.path.join(folder, '../../data/General100/')
    in_dir = data_dir + '/original'
    out_dir = data_dir + '/FSRCNN'
    
    generate_dataset(in_dir, out_dir, 3, 7, 7)
    
    




