from os import listdir
from os.path import join

from PIL import Image, ImageOps
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision import transforms as transform

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


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
