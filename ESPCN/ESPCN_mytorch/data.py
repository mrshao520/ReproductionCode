from os import listdir
from os.path import join
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageOps

def is_image_file(filename):
    return any(filename.endswith(extension) 
                for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])

def is_video_file(filename):
    return any(filename.endswith(extension) 
               for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])

class DatasetFromFolder(Dataset):
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