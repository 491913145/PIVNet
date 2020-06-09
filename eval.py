from torchsummary import summary
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
import os
from models.PWCNet import pwc_dc_net
import torch
from torch.autograd import Variable
from utils.multiscaleloss import realEPE,RMSE
from tqdm import tqdm
from utils.dataloader import MyDataset
from utils.augmentations import Basetransform
from utils.flowlib import *
import torch.nn.functional as F

__all__ = [
    'eval'
]


def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new




def eval(model,ImgLoader):
    total_test_rmse = []
    iterator = iter(ImgLoader)
    step = len(ImgLoader)
    # step = 50
    model.eval()
    print('evaluating... ')
    for i in tqdm(range(step)):
        img1, img2, flo = next(iterator)
        img1 = Variable(torch.FloatTensor(img1.float()))
        img2 = Variable(torch.FloatTensor(img2.float()))
        flo = Variable(torch.FloatTensor(flo.float()))
        imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
        output = model((imgL, imgR))
        total_test_rmse += [RMSE(output.detach(), flowl0.detach()).cpu().numpy()]
    return np.mean(total_test_rmse)


