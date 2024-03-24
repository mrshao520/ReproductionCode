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
    if os.path.exists(out_dir):
        print(f'the train data exists!!!')
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
    
    # 记录总数
    count = 1
    # 目标图像大小
    target_size = size_input * upscale_factor
    for image_name in images_name:
        input = Image.open(in_dir + '/' + image_name)
        target = input.copy()
        
        input = ImageOps.scale(input, 1/upscale_factor, 
                               resample=Image.Resampling.BICUBIC)
        in_width = input.width
        in_height = input.height
        for x in range(0, in_width - size_input + 1, stride):
                for y in range(0, in_height - size_input + 1, stride):
                    sub_input = input.crop((x, y, x + size_input, 
                                                 y + size_input))
                    
                    tar_x = x * upscale_factor
                    tar_y = y * upscale_factor
                    sub_target = target.crop((tar_x, tar_y, tar_x + target_size,
                                                 tar_y + target_size))
                    
                    sub_input.save(f'{input_path}/{count}_{image_name}')
                    sub_target.save(f'{target_path}/{count}_{image_name}')
                    count = count + 1
                    
    print(f'the number of images: {count}')

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor) -> None:
        super().__init__()
        self.hr_path = dataset_dir + '/HR'
        self.lr_path = dataset_dir + '/LR'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
        self.totensor = transform.ToTensor()

    def __getitem__(self, index):
        lr_image = Image.open(self.lr_filenames[index])
        hr_image = Image.open(self.hr_filenames[index])
        return self.totensor(lr_image), self.totensor(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor) -> None:
        super().__init__()
        self.hr_path = dataset_dir + '/HR'
        self.lr_path = dataset_dir + '/LR'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
        self.totensor = transform.ToTensor()

    def __getitem__(self, index):
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_restore_img = ImageOps.scale(lr_image, self.upscale_factor, 
                                        resample=Image.BICUBIC)
        return self.totensor(lr_image), self.totensor(hr_restore_img), self.totensor(hr_image)
        

    def __len__(self):
        return len(self.lr_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor) -> None:
        super().__init__()
        self.hr_path = dataset_dir + '/HR'
        self.lr_path = dataset_dir + '/LR'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
        self.totensor = transform.ToTensor()
        
    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_restore_img = ImageOps.scale(lr_image, self.upscale_factor, 
                                        resample=Image.BICUBIC)
        return image_name, self.totensor(lr_image), self.totensor(hr_restore_img), self.totensor(hr_image)
        
    def __len__(self):
        return len(self.lr_filenames)




        

if __name__ == '__main__':
    folder = os.path.dirname(__file__)
    data_dir = os.path.join(folder, '../data/VOC2012/')
    train_in_dir = data_dir + '/train'
    train_out_dir = data_dir + '/X4/train'
    
    val_in_dir = data_dir + '/val'
    val_out_dir = data_dir + '/X4/val'
    
    generate_dataset(train_in_dir, train_out_dir, 4, 24, 24)
    generate_dataset(val_in_dir, val_out_dir, 4, 24, 24)
    
    




