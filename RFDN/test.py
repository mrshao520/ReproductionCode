import os.path
import logging
import time
from collections import OrderedDict
import torch

from utils import utils_logger
from utils import utils_image as util
from RFDN import RFDN



def main():
    current_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    current_path = current_path.replace('\\', '/')  
    data_path = current_path + '../data/'
    print(f'current path {current_path} \n data_path : {data_path}')

    utils_logger.logger_info('AIM-track', log_path=f'{current_path}AIM-track.log')
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = data_path + 'Set5'
    testset_L = 'X4'
    #testset_L = 'DIV2K_test_LR_bicubic'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    #torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join(f'{current_path}/trained_model', 'RFDN_AIM.pth')
    model = RFDN()
    # print(f'model : {model}')
    load_net = torch.load(model_path)
    # print(f'load_net : {load_net.keys()}')
    model.load_state_dict(load_net, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L, 'LR')
    E_folder = os.path.join(current_path, testset_L+'_results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    img_SR = []
    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        print(f'img_L shape : {img_L.shape}')
        img_L = img_L.to(device)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        img_SR.append(img_E)

        # --------------------------------
        # (3) save results
        # --------------------------------
        util.imsave(img_E, os.path.join(E_folder, img_name+ext))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

    # --------------------------------
    # (4) calculate psnr
    # --------------------------------
    
    psnr = []
    idx = 0
    H_folder = os.path.join(testsets, testset_L, 'HR')

    for img in util.get_image_paths(H_folder):
        img_H = util.imread_uint(img, n_channels=3)
        psnr.append(util.calculate_psnr(img_SR[idx], img_H, 4))
        idx += 1
    logger.info(util.get_image_paths(L_folder))
    logger.info(f'psnr : {psnr}')
    logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))
    

if __name__ == '__main__':

    main()
