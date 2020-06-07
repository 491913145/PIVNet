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
        nn.LeakyReLU(0.1,inplace=True))


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

        self.conv1a = conv(1, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

        self.corr = Correlation()
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        od = nd

        self.dc_conv1 = conv(od, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(od, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(od, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(od, 128, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(od, 128, kernel_size=3, stride=1, padding=16, dilation=16)

        od = 196*2

        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd + 128 + 4
        od = 260
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(2 + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(2 + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(2 + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5_1 = predict_flow(od + dd[1])
        self.predict_flow5_2 = predict_flow(450)
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(450, 2, kernel_size=4, stride=2, padding=1)

        # od = nd + 96 + 4
        od = 196
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(-62 + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(-62 + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(-62 + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4_1 = predict_flow(od + dd[1])
        self.predict_flow4_2 = predict_flow(-62 + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(-62 + dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd + 64 + 4
        od = 132
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(-126 + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(-126 + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(-126 + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3_1 = predict_flow(od + dd[1])
        self.predict_flow3_2 = predict_flow(-126 + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(-126 + dd[4], 2, kernel_size=4, stride=2, padding=1)

        # od = nd + 32 + 4
        od = 68
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(-190 + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(-190 + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(-190 + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2_1 = predict_flow(od + dd[1])
        self.predict_flow2_2 = predict_flow(-190 + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = deconv(-190 + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 16 + 4
        self.conv1_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv1_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv1_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv1_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv1_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow1_1 = predict_flow(od + dd[1])
        self.predict_flow1_2 = predict_flow(od + dd[2])

        self.dc_conv7 = predict_flow(32)

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

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        # corr6 = self.corr(c16, c26)
        # corr6 = self.leakyRELU(corr6)
        # corr6 = self.dc_conv1(corr6) + self.dc_conv2(corr6) + self.dc_conv3(corr6) + self.dc_conv4(
        #     corr6) + self.dc_conv5(corr6)
        corr6 = torch.cat((c16, c26),1)

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6)
        # corr5 = self.corr(c15, warp5)
        # corr5 = self.leakyRELU(corr5)
        x = torch.cat((warp5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        flow5 = self.predict_flow5_1(x) + up_flow6
        warp5 = self.warp(c25, flow5)
        x = torch.cat((warp5, c15, flow5), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5_2(x) + flow5
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5)
        # corr4 = self.corr(c14, warp4)
        # corr4 = self.leakyRELU(corr4)
        x = torch.cat((warp4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        flow4 = self.predict_flow4_1(x) + up_flow5
        warp4 = self.warp(c24, flow4)
        x = torch.cat((warp4, c14, flow4), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4_2(x) + flow4
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4)
        # corr3 = self.corr(c13, warp3)
        # corr3 = self.leakyRELU(corr3)
        x = torch.cat((warp3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        flow3 = self.predict_flow3_1(x) + up_flow4
        warp3 = self.warp(c23, flow3)
        x = torch.cat((warp3, c13,flow3), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3_2(x) + flow3
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3)
        # corr2 = self.corr(c12, warp2)
        # corr2 = self.leakyRELU(corr2)
        x = torch.cat((warp2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        flow2 = self.predict_flow2_1(x) + up_flow3
        warp2 = self.warp(c22, flow2)
        x = torch.cat((warp2, c12, flow2), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2_2(x) + flow2

        # up_flow2 = self.deconv2(flow2)
        # up_feat2 = self.upfeat2(x)
        #
        # warp1 = self.warp(c21, up_flow2)
        # corr1 = self.corr(c11, warp1)
        # corr1 = self.leakyRELU(corr1)
        # x = torch.cat((corr1, c11, up_flow2, up_feat2), 1)
        # x = torch.cat((self.conv1_0(x), x), 1)
        # x = torch.cat((self.conv1_1(x), x), 1)
        # up_flow2 = self.predict_flow1_1(x) + up_flow2
        # warp1 = self.warp(c21, up_flow2)
        # x = torch.cat((warp1, c11, up_flow2), 1)
        # x = torch.cat((self.conv1_0(x), x), 1)
        # x = torch.cat((self.conv1_1(x), x), 1)
        # x = torch.cat((self.conv1_2(x), x), 1)
        # flow1 = self.predict_flow1_2(x) + up_flow2

        # x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        # flow1 = flow1 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2


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
