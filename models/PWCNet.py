"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os

os.environ['PYTHON_EGG_CACHE'] = 'tmp/'  # a writable directory
from correlation.correlation import ModuleCorrelation as Correlation
import numpy as np

__all__ = [
    'pwc_dc_net'
]


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet, self).__init__()

        channel = [3,16,16,16,32,32,32,64,64,64,96,96,96,128,128,128,196,196,196]

        self.netC = []
        for i in range(len(channel)-1):
            self.netC += [conv(channel[i], channel[i+1], kernel_size=3, stride=1 if i%3 else 2)]
        self.netC = nn.ModuleList(self.netC)

        self.corr = Correlation()
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.array([128, 128, 96, 64, 32])

        self.netE = []

        od = nd
        for i in range(5):
            block = []
            block += [conv(od, dd[0], kernel_size=3, stride=1)]
            block += [conv(dd[0], dd[1], kernel_size=3, stride=1)]
            block += [conv(dd[1], dd[2], kernel_size=3, stride=1)]
            block += [conv(dd[2], dd[3], kernel_size=3, stride=1)]
            block += [conv(dd[3], dd[4], kernel_size=3, stride=1)]
            block += [predict_flow(dd[4])]
            block += [deconv(2, 2, kernel_size=4, stride=2, padding=1)]
            block += [deconv(dd[4], 2, kernel_size=4, stride=2, padding=1)]
            self.netE += [nn.ModuleList(block)]
        self.netE = nn.ModuleList(self.netE)

        # self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        # self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, x):
        im1 = x[0]
        im2 = x[1]

        feats = []
        for i in range(0,len(self.netC),3):
            im1 = self.netC[i](im1)
            im1 = self.netC[i+1](im1)
            im1 = self.netC[i+2](im1)
            im2 = self.netC[i](im2)
            im2 = self.netC[i+1](im2)
            im2 = self.netC[i+2](im2)
            feats += [im1,im2]

        feats = feats[::-1]
        corr = self.corr(feats[0], feats[1])
        corr = self.leakyRELU(corr)
        ratio = [0.625,1.25,2.5,5,10]
        flows = []
        for i in range(len(self.netE)):
            x = corr
            for c in self.netE[i][:-2]:
                x = c(x)
            flows += [x]
            up_flow = self.netE[i][-2](x)
            warp = self.warp(feats[2*(i+1)+1], up_flow * ratio[i])
            corr = self.corr(feats[2*(i+1)], warp)
            corr = self.leakyRELU(corr)

        if self.training:
            return flows[::-1]
        else:
            return flows[-1]


def pwc_dc_net(path=None):
    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            # model_dict = model.state_dict()
            # pretrained_dict  = {k: v for k, v in model_dict.items() if k in data['state_dict']}
            # model_dict.update(pretrained_dict)
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model
