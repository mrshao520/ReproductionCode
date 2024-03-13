import argparse
import os, sys
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize
from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswith(extension) 
                for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

def is_video_file(filename):
    return any(filename.endswith(extension) 
               for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])

# 设置crop_size是upscale_factor的整数倍
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# 裁剪图片并缩小upscale_factor倍, 
def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])

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
        
    def __gettime__(self, index):
        input, _, _ = Image.open(self.input_filenames[index]).convert('YCbCr').split()
        target, _, _ = Image.open(self.target_filenames[index]).convert('YCbCr').split()
        
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            
        return input, target
    
    def __len__(self):
        return len(self.input_filenames)
        
def generate_dataset(image_dir, data_type, upscale_factor):
    path = image_dir + '/X' + str(upscale_factor) + '/' + data_type
    if os.path.exists(path):
        return
    
    images_name = [x for x in listdir(image_dir + '/' + data_type) 
                   if is_image_file(x)]
    
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)
    
    path = image_dir + '/X' + str(upscale_factor) + '/' + data_type
    if not os.path.exists(path):
        os.makedirs(path)
    input_path = path + '/LR'
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    target_path = path + '/HR'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        
    for image_name in tqdm(images_name, desc='generate '+data_type+' dataset with upscale factor = '
                           + str(upscale_factor) + ' from VOC2012'):
        image = Image.open(image_dir + '/' + data_type + '/' + image_name)
        target = image.copy()
        
        image = lr_transform(image)
        target = hr_transform(target)
        
        image.save(input_path + '/' + image_name)
        target.save(target_path + '/' + image_name)
    

def generate_dataset(image_dir, upscale_factor):
    path = image_dir + '/X' + str(upscale_factor)
    if os.path.exists(path):
        return
    
    images_name = [x for x in listdir(image_dir + '/' + 'original') 
                   if is_image_file(x)]
    
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size)
    
    path = image_dir + '/X' + str(upscale_factor) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    input_path = path + '/LR'
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    target_path = path + '/HR'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        
    for image_name in tqdm(images_name, desc='generate  dataset with upscale factor = '
                           + str(upscale_factor) + ' from VOC2012'):
        image = Image.open(image_dir  + '/' + image_name)
        target = image.copy()
        
        image = lr_transform(image)
        target = hr_transform(target)
        
        image.save(input_path + '/' + image_name)
        target.save(target_path + '/' + image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    
    current_path, filename = os.path.split(os.path.abspath(sys.argv[0]))
    current_path = current_path.replace('\\', '/')
    image_dir = os.path.join(current_path, '../../data/DSDS200/')
    print(f'image_data:{image_dir}')
    generate_dataset(image_dir, upscale_factor=UPSCALE_FACTOR)
