## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
# from utils.interpolation import Interp2, Interp2MaskBinary
# from utils.interpolation import Meshgrid
import numpy as np
import cv2


def denormalize_coords(xx, yy, width, height):
    """ scale indices from [-1, 1] to [0, width/height] """
    xx = 0.5 * (width - 1.0) * (xx.float() + 1.0)
    yy = 0.5 * (height - 1.0) * (yy.float() + 1.0)
    return xx, yy


def normalize_coords(xx, yy, width, height):
    """ scale indices from [0, width/height] to [-1, 1] """
    xx = (2.0 / (width - 1.0)) * xx.float() - 1.0
    yy = (2.0 / (height - 1.0)) * yy.float() - 1.0
    return xx, yy


def apply_transform_to_params(theta0, theta_transform):
    a1 = theta0[:, 0]
    a2 = theta0[:, 1]
    a3 = theta0[:, 2]
    a4 = theta0[:, 3]
    a5 = theta0[:, 4]
    a6 = theta0[:, 5]
    #
    b1 = theta_transform[:, 0]
    b2 = theta_transform[:, 1]
    b3 = theta_transform[:, 2]
    b4 = theta_transform[:, 3]
    b5 = theta_transform[:, 4]
    b6 = theta_transform[:, 5]
    #
    c1 = a1 * b1 + a4 * b2
    c2 = a2 * b1 + a5 * b2
    c3 = b3 + a3 * b1 + a6 * b2
    c4 = a1 * b4 + a4 * b5
    c5 = a2 * b4 + a5 * b5
    c6 = b6 + a3 * b4 + a6 * b5
    #
    new_theta = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)
    return new_theta


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, flo):
        for t in self.transforms:
            img1, img2, flo = t(img1, img2, flo)
        return img1, img2, flo


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img1, img2, flo):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        flo = flo.astype(np.float32)
        img1 -= self.mean
        img2 -= self.mean
        return img1, img2, flo


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img1, img2, flo):
        img1 = cv2.resize(img1, (self.size, self.size))
        img2 = cv2.resize(img2, (self.size, self.size))
        flo = cv2.resize(flo, (self.size, self.size))
        return img1, img2, flo


class ConvertGray(object):
    def __init__(self):
        pass

    def __call__(self, img1, img2, flo):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        return img1, img2, flo


class Pemute(object):
    def __init__(self, order=(2, 0, 1)):
        self.order = order

    def __call__(self, img1, img2, flo):
        img1 = img1.transpose(self.order)
        img2 = img2.transpose(self.order)
        flo = flo.transpose(self.order)
        return img1, img2, flo


class RandomMirror(object):
    def __call__(self, img1, img2, flo):
        if np.random.randint(2):
            img1 = img1[:, ::-1].copy()
            img2 = img2[:, ::-1].copy()
            flo = flo[:, ::-1].copy()
            flo[:, :, 0] = -flo[:, :, 0]
        if np.random.randint(2):
            img1 = img1[::-1, :].copy()
            img2 = img2[::-1, :].copy()
            flo = flo[::-1, :].copy()
            flo[:, :, 1] = -flo[:, :, 1]
        return img1, img2, flo


class Scale(object):
    def __init__(self, scale=128):
        self.scale = scale

    def __call__(self, img1, img2, flo):
        img1 = img1 / self.scale
        img2 = img2 / self.scale
        return img1, img2, flo


class RandomSampleCrop(object):
    def __call__(self, img1, img2, flo):
        height, width, _ = img1.shape
        if np.random.randint(2):
            return img1, img2, flo
        w = np.random.uniform(0.3 * width, width)
        h = np.random.uniform(0.3 * height, height)

        left = np.random.uniform(width - w)
        top = np.random.uniform(height - h)

        # convert to integer rect x1,y1,x2,y2
        rect = np.array([int(left), int(top), int(left + w), int(top + h)])

        # cut the crop from the image
        img1 = img1[rect[1]:rect[3], rect[0]:rect[2], :]
        img2 = img2[rect[1]:rect[3], rect[0]:rect[2], :]
        flo = flo[rect[1]:rect[3], rect[0]:rect[2], :]

        return img1, img2, flo


class Expand(object):
    def __call__(self, img1, img2, flo):
        if np.random.randint(2):
            return img1, img2, flo

        height, width, depth = img1.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_img1 = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=img1.dtype)
        expand_img1[int(top):int(top + height), int(left):int(left + width)] = img1
        img1 = expand_img1

        expand_img2 = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=img2.dtype)
        expand_img2[int(top):int(top + height), int(left):int(left + width)] = img2
        img2 = expand_img2

        expand_flo = np.zeros((int(height * ratio), int(width * ratio), flo.shape[2]), dtype=flo.dtype)
        expand_flo[int(top):int(top + height), int(left):int(left + width)] = flo
        flo = expand_flo

        return img1, img2, flo


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img1, img2, flo):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            img1 = img1 + delta
            img2 = img2 + delta
        return img1, img2, flo


class RandomRotate(object):
    def __call__(self, img1, img2, flo):
        mode = np.random.randint(3)
        if mode == 0:
            img1 = np.rot90(img1)
            img2 = np.rot90(img2)
            flo = np.rot90(flo)
            flo = flo[:, :, ::-1]
            flo[:, :, 1] = -flo[:, :, 1]
        elif mode == 1:
            img1 = np.rot90(img1[::-1,::-1,:])
            img2 = np.rot90(img2[::-1,::-1,:])
            flo = np.rot90(flo[::-1,::-1,:])
            flo = flo[:, :, ::-1]
            flo[:, :, 0] = -flo[:, :, 0]
        return img1.copy(),img2.copy(),flo.copy()

class RandomGasuss(object):
    def __init__(self, mean=0, var=0.001):
        self.mean=mean
        self.var = var
    def __call__(self, img1, img2, flo):
        cv2.namedWindow('s',cv2.WINDOW_FREERATIO)
        cv2.imshow('s',img1)
        cv2.waitKey()
        img1 = img1.astype(np.float)/255
        img2 = img2.astype(np.float)/255
        noise = np.random.normal(self.mean, self.var, img1.shape)
        img1 = img1 + noise
        img1 = (img1*255).astype(np.uint8)
        noise = np.random.normal(self.mean, self.var, img2.shape)
        img2 = img2 + noise
        img2 = (img2*255).astype(np.uint8)
        img1 = np.clip(img1, 0, 255)
        img2 = np.clip(img2, 0, 255)
        cv2.imshow('s',img1)
        cv2.waitKey()
        return img1, img2, flo

class Augmentation(object):
    def __init__(self, size=256, mean=(128)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # ConvertGray(),
            RandomBrightness(),
            RandomRotate(),
            RandomMirror(),
            Expand(),
            RandomSampleCrop(),
            Resize(self.size),
            SubtractMeans(self.mean),
            # Scale(),
            Pemute(),
        ])

    def __call__(self, img1, img2, flo):
        return self.augment(img1, img2, flo)


class Basetransform(object):
    def __init__(self, size=256, mean=(128)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            # ConvertGray(),
            Resize(self.size),
            SubtractMeans(self.mean),
            # Scale(),
            Pemute()
        ])

    def __call__(self, img1, img2, flo):
        return self.augment(img1, img2, flo)
