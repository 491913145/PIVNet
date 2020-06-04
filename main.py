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
torch.backends.cudnn.benchmark = True
from utils.multiscaleloss import MultiscaleLoss, realEPE
from glob import glob
import cv2 as cv
from tqdm import tqdm

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
batch_size = 32

torch.cuda.set_device(1)






dataset = MyDataset('/home/disk/lihaiyun/LiteFlow/PIV-LiteFlowNet-en/lite/dataset/splits/train', shape=(256, 256))

print('%d batches per epoch' % (len(dataset) // batch_size))

from models.PWCNet import pwc_dc_net

model = pwc_dc_net(args.resume)
# model = nn.DataParallel(model) #就单GPU走一波吧
model.cuda()
summary(model, input_size=(1, 1, 256, 256))

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items()}

    model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    if args.retrain == 'true':
        print('re-training')
    else:
        with open('./iter_counts-%d.txt' % int(args.logname.split('-')[-1]), 'r') as f:
            total_iters = int(f.readline())
        print('resuming from %d' % total_iters)
        mean_L = pretrained_dict['mean_L']
        mean_R = pretrained_dict['mean_R']

# print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), amsgrad=False)
criterion = MultiscaleLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)


def train(imgL, imgR, flowl0):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL.float()))
    imgR = Variable(torch.FloatTensor(imgR.float()))
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
    vis['AEPE'] = realEPE(output[0].detach(), flowl0.detach())
    return loss.data, vis


def main():
    TrainImgLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size,
                                                 drop_last=True, pin_memory=True)
    log = logger.Logger(args.savemodel, name=args.logname)

    start_full_time = time.time()
    total_iters = 0
    start_epoch = 1 if args.resume is None else int(re.findall('(\d+)',args.resume)[0])+1
    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_train_aepe = 0

        # training loop
        for batch_idx, (imgL_crop, imgR_crop, flowl0) in enumerate(TrainImgLoader):

            imgL_crop /= 255
            imgR_crop /= 255

            start_time = time.time()
            loss, vis = train(imgL_crop, imgR_crop, flowl0)
            if (total_iters + 1) % 20 == 0:
                print('Iter %d training loss = %.3f , AEPE = %.3f , time = %.2f' % (
                total_iters + 1, loss, vis['AEPE'], time.time() - start_time))
            total_train_loss += loss
            total_train_aepe += vis['AEPE']
            total_iters += 1

        savefilename = args.savemodel + '/' + args.logname + '/finetune_' + str(total_iters) + '.tar'
        save_dict = model.state_dict()
        save_dict = collections.OrderedDict(
            {k: v for k, v in save_dict.items() if ('flow_reg' not in k or 'conv1' in k) and ('grid' not in k)})
        torch.save(
            {'iters': total_iters, 'state_dict': save_dict, 'train_loss': total_train_loss / len(TrainImgLoader), },
            savefilename)
        log.scalar_summary('train/loss', total_train_loss / len(TrainImgLoader), epoch)
        log.scalar_summary('train/aepe', total_train_aepe / len(TrainImgLoader), epoch)
        scheduler.step(total_train_loss / len(TrainImgLoader))

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))


if __name__ == '__main__':
    main()
