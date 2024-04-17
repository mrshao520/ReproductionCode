import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from BSRN_arch import BSRN
import torch


def image2tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    totensor = transforms.ToTensor()
    image_tensor = totensor(image)
    return image_tensor

image_path = r"D:\home\ReproductionCode\BSRN\monarch.png"
model_path = r"D:\home\ReproductionCode\BSRN\pretrained_models\net_g_BSRN_x4.pth"
image_lr = image2tensor(image_path)
image_lr = image_lr.unsqueeze(0)

model = BSRN()
load_net = torch.load(model_path)
model.load_state_dict(load_net['params'], strict=True)
model = model.eval()

with torch.no_grad():
    image_sr = model(image_lr)



