import argparse, os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import DatasetFromFolder
from model import SRCNN

parser = argparse.ArgumentParser(description='SRCNN training parameters')
parser.add_argument('--zoom_factor', type=int, required=True)
parser.add_argument('--nb_epochs', type=int, default=200)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

print(args.zoom_factor)

device = torch.device('cuda:0' if (torch.cuda.is_available() and args.cuda) else 'cpu')

'''
PyTorch中用于设置随机数生成器的种子，以确保实验的可重复性

如果您希望在多次运行程序时得到相同的结果，或者在进行比较实验时保证条件一致，设置种子是一个很好的做法。
不过，如果您追求每次运行时的随机性，就不需要设置种子或者设置不同的种子。
'''
torch.manual_seed(0)  # 设置PyTorch中所有随机数生成器的种子为0
torch.cuda.manual_seed(0) # 设置使用CUDA时的随机数生成器的种子为0

# Parameters
BATCH_SIZE = 4
NUM_WORKERS = 0  # on Windows, set this variable to 0

folder = os.path.dirname(__file__)
train_path = os.path.join(folder, './data/train')
test_path = os.path.join(folder, './data/test')
trainset = DatasetFromFolder(train_path, zoom_factor=args.zoom_factor)
testset = DatasetFromFolder(test_path, zoom_factor=args.zoom_factor)

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = SRCNN().to(device)
criterion = nn.MSELoss() # 损失函数
optimizer = optim.Adam(
    [
        {'params': model.conv1.parameters(), 'lr': 0.0001},
        {'params': model.conv2.parameters(), 'lr': 0.0001},
        {'params': model.conv3.parameters(), 'lr': 0.00001}
    ], lr=0.00001
)

for epoch in range(args.nb_epochs):
    # Train
    epoch_loss = 0
    model.train()
    for iteration, (X, y) in enumerate(trainloader):
        # 输入，标签
        input, target = X.to(device), y.to(device)
        
        # 前向传播
        out = model(input)
        # 计算损失
        loss = criterion(out, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()
        # loss是一个张量，使用item获取真实值
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch}. Training loss: {epoch_loss / len(trainloader)}')
    
    # Test
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for X, y in testloader:
            input, target = X.to(device), y.to(device)
            
            out = model(input)
            loss = criterion(out, target)
            psnr = 10 * log10(1 / loss.item())
            avg_psnr += psnr
            
    print(f'Average PSNR: {avg_psnr / len(testloader)} dB.')
    
# save model
torch.save(model, f'SRCNN_model.pth')
            





