import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from data import load_img, CROP_SIZE

parser = argparse.ArgumentParser(description='SRCNN run parameters')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

img = Image.open(args.image).convert('YCbCr')
img = img.resize((int(img.size[0] * args.zoom_factor), int(img.size[1]*args.zoom_factor)), Image.BICUBIC)
y, Cb, Cr = img.split()

img_to_tensor = transforms.ToTensor()
input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')
print(device)
model = torch.load(args.model).to(device)
input = input.to(device)

out = model(input)
out = out.cpu()

out_img_y = out[0].detach().numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img = Image.merge('YCbCr', [out_img_y, Cb, Cr]).convert('RGB')
out_img.save(f'zoomed_{args.image}')



