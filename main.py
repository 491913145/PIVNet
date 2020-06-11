from __future__ import print_function
import cv2

cv2.setNumThreads(0)
import sys
import pdb
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import re
from utils.flowlib import flow_to_image
from utils import logger
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataloader import MyDataset
from utils.augmentations import Augmentation, Basetransform
torch.backends.cudnn.benchmark = True
from utils.multiscaleloss import MultiscaleLoss, realEPE, RMSE
from glob import glob
import cv2 as cv
from tqdm import tqdm
from eval import eval


def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float, default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size')
parser.add_argument('--logname', default='logname',
                    help='name of the log file')
parser.add_argument('--database', default='/',
                    help='path to the database')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model')
parser.add_argument('--savemodel', default='./',
                    help='path to save the model')
parser.add_argument('--resume', default=None,
                    help='whether to reset moving mean / other hyperparameters')
parser.add_argument('--stage', default='chairs',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval}')
parser.add_argument('--ngpus', type=int, default=2,
                    help='number of gpus to use.')
args = parser.parse_args()

baselr = 1e-3
batch_size = 16

torch.cuda.set_device(1)

dataset = MyDataset('/home/disk/lihaiyun/LiteFlow/PIV-LiteFlowNet-en/lite/dataset/splits/train',
                    transform=Augmentation(size=256, mean=(128)))
test_dataset = MyDataset('/home/disk/lihaiyun/LiteFlow/PIV-LiteFlowNet-en/lite/dataset/splits/test',
                         transform=Basetransform(size=256, mean=(128)))

print('%d batches per epoch' % (len(dataset) // batch_size))

from models.PWCNet import pwc_dc_net

model = pwc_dc_net(args.resume)
# model = nn.DataParallel(model) #就单GPU走一波吧
model.cuda()
summary(model, input_size=(2, 3, 256, 256))

optimizer = optim.Adam(model.parameters(), lr=baselr, betas=(0.9, 0.999), amsgrad=False)
# optimizer = optim.SGD(model.parameters(), lr=baselr, momentum=0.9,  weight_decay=5e-4)
criterion = MultiscaleLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True, num_workers=12,
                                            drop_last=True, pin_memory=True)


def train(imgL, imgR, flowl0):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL.float()))
    imgR = Variable(torch.FloatTensor(imgR.float()))
    with torch.no_grad():
        flowl0 = Variable(torch.FloatTensor(flowl0.float()))

    imgL, imgR, flowl0 = imgL.cuda(), imgR.cuda(), flowl0.cuda()
    # forward-backward
    optimizer.zero_grad()
    output = model((imgL, imgR))
    loss = criterion.forward(output, flowl0)
    loss.backward()
    optimizer.step()

    vis = {}
    vis['output2'] = output[0].detach().cpu().numpy()
    vis['output3'] = output[1].detach().cpu().numpy()
    vis['output4'] = output[2].detach().cpu().numpy()
    vis['output5'] = output[3].detach().cpu().numpy()
    vis['output6'] = output[4].detach().cpu().numpy()
    vis['RMSE'] = RMSE(output[0].detach(), flowl0.detach())
    return loss.data, vis


def main():
    TrainImgLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                 drop_last=True, pin_memory=True)
    log = logger.Logger(args.savemodel, name=args.logname)

    start_full_time = time.time()
    start_epoch = 1 if args.resume is None else int(re.findall('(\d+)', args.resume)[0]) + 1
    total_iters = 0
    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_train_rmse = 0
        # training loop
        for batch_idx, (imgL_crop, imgR_crop, flowl0) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss, vis = train(imgL_crop, imgR_crop, flowl0)
            if (total_iters + 1) % 20 == 0:
                print('Epoch %d Iter %d/%d training loss = %.3f , RMSE = %.3f , time = %.2f' % (epoch,
                                                                                             batch_idx,len(TrainImgLoader), loss,
                                                                                             vis['RMSE'],
                                                                                             time.time() - start_time))
            total_train_loss += loss
            total_train_rmse += vis['RMSE']
            total_iters += 1
        savefilename = args.savemodel + '/' + args.logname + '/finetune_' + str(epoch) + '.tar'
        save_dict = model.state_dict()
        save_dict = collections.OrderedDict(
            {k: v for k, v in save_dict.items() if ('flow_reg' not in k or 'conv1' in k) and ('grid' not in k)})
        torch.save(
            {'epoch': epoch, 'state_dict': save_dict, 'train_loss': total_train_loss / len(TrainImgLoader), },
            savefilename)
        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        log.scalar_summary('train/RMSE', total_train_rmse / len(TrainImgLoader), epoch)
        log.scalar_summary('test/RMSE', eval(model, TestImgLoader), epoch)
        log.scalar_summary('train/learning rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step(total_train_loss / len(TrainImgLoader))

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()
