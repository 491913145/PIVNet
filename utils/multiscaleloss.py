"""
Taken from https://github.com/ClementPinard/FlowNetPytorch
"""
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
from torch.autograd import Variable
import numpy as np


# def rob_EPE(input_flow, target_flow, mask, sparse=False, mean=True):
#     #mask = target_flow[:,2]>0
#     target_flow = target_flow[:,:2]
#     #TODO
# #    EPE_map = torch.norm(target_flow-input_flow,2,1)
#     EPE_map = (torch.norm(target_flow-input_flow,1,1)+0.01).pow(0.4)
#     batch_size = EPE_map.size(0)
#     if sparse:
#         # invalid flow is defined with both flow coordinates to be exactly 0
#         mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
#
#         EPE_map = EPE_map[~mask]
#     if mean:
#         return EPE_map[mask].mean()
#     else:
#         return EPE_map[mask].sum()/batch_size

# def sparse_max_pool(input, size):
#     '''Downsample the input by considering 0 values as invalid.
#
#     Unfortunately, no generic interpolation mode can resize a sparse map correctly,
#     the strategy here is to use max pooling for positive values and "min pooling"
#     for negative values, the two results are then summed.
#     This technique allows sparsity to be minized, contrary to nearest interpolation,
#     which could potentially lose information for isolated data points.'''
#
#     positive = (input > 0).float()
#     negative = (input < 0).float()
#     output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
#     return output

loss_fn = nn.MSELoss(reduce=False,size_average=False)

def EPE(input_flow, target_flow, mean=True):
    # EPE_map = torch.sqrt(torch.norm(target_flow - input_flow, 2, 1))
    # EPE_map = F.smooth_l1_loss(input_flow,target_flow,reduce=False,size_average=False)
    EPE_map = loss_fn(input_flow,target_flow)
    batch_size = EPE_map.size(0)
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum() / batch_size


def RMSE(input_flow, target_flow):
    b, _, h, w = target_flow.size()
    input_flow = F.interpolate(input_flow, (h, w), mode='bilinear', align_corners=False)
    return torch.sqrt((target_flow - input_flow).pow(2).mean())


def multiscaleEPE(network_output, target_flow, weights=(0.005,0.005, 0.01, 0.02, 0.08, 0.32)):
    def one_scale(output, target):
        b, _, h, w = output.size()
        target_scaled = F.interpolate(target, (h, w), mode='area')
        return EPE(output, target_scaled, mean=False)
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    assert (len(weights) == len(network_output))
    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, mean=True)


class MultiscaleLoss(nn.Module):
    def __init__(self):
        super(MultiscaleLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predictions, targets):
        return multiscaleEPE(predictions, targets)
