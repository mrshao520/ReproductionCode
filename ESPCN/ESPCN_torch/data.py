import argparse
import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, CenterCrop, Scale
from tqdm import tqdm

def is_image_file(filename):
    return any(filename.endswitch(extension) 
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
        Scale(crop_size // upscale_factor, interpolation=Image.BICUBIC)
    ])

def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size)
    ])

class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, 
                 input_transform=None, target_transform=None) -> None:
        super().__init__()

