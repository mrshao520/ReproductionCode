clear;close all;
%% settings
%% 设置文件夹路径、保存路径、输入和标签的大小、缩放和步长。
folder = 'Train';
savepath = 'train.h5';
size_input = 33; % 输入图像的大小
size_label = 21; % 标签图像的大小
scale = 3; % 下采样因子
stride = 14; % 步长

%% initialization
%% 初始化数据和标签矩阵，并计算填充大小。
data = zeros(size_input, size_input, 1, 1); % 初始化数据矩阵
label = zeros(size_label, size_label, 1, 1); % 初始化标签矩阵
padding = abs(size_input - size_label)/2; % 计算填充大小
count = 0; % 初始化计数器

%% generate data
%% 遍历文件夹中的所有BMP图像文件，将其转换为YCbCr格式，并将其调整为所需的输入和标签大小。

filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));  % 读取图像
    image = rgb2ycbcr(image); % 转换图像颜色空间到YCbCr
    %% image(:, :, 1)选取了Y通道
    image = im2double(image(:, :, 1)); % 将图像转换为浮点数
    
    im_label = modcrop(image, scale); % 将图片裁剪为能够调整的大小（与放大率匹配）。裁剪舍掉余数行和列。
    [hei,wid] = size(im_label); % 获取裁剪后图像的大小
    %% 先将图片下采样，后使用双三次插值上采样
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic'); 

    %% im_label: super resulotion image
    %% im_input: lower resulotion image
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1); % 提取输入图像的子区域
            %% 提取标签图像的子区域
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1; % 增加计数器
            data(:, :, 1, count) = subim_input; % 存储子图像数据
            label(:, :, 1, count) = subim_label; % 存储子图像标签
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
