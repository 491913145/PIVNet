from torch.utils.data import Dataset
import cv2 as cv
import numpy as np
from glob import glob
import os


def readFlowFile(filename):
    import struct
    import numpy as np
    fid = open(filename, 'rb')
    tag = struct.unpack('f', fid.read(4))[0]
    width = struct.unpack('i', fid.read(4))[0]
    height = struct.unpack('i', fid.read(4))[0]
    img = []
    while True:
        try:
            data = struct.unpack('f', fid.read(4))[0]
            img.append(data)
        except:
            break
    img = np.reshape(img, (width, height, -1))
    return img


def load_data(path):
    flo_paths = glob(os.path.join(path, '*.flo'))
    img0_paths = [x.replace('flow.flo', 'img1.tif') for x in flo_paths]
    img1_paths = [x.replace('flow.flo', 'img2.tif') for x in flo_paths]
    return flo_paths, img0_paths, img1_paths


class MyDataset(Dataset):
    def __init__(self, path,  transform=None):
        self.flo_paths, self.img0_paths, self.img1_paths = load_data(path)
        self.transform = transform

    def __getitem__(self, i):
        img1 = cv.imread(self.img0_paths[i])
        img2 = cv.imread(self.img1_paths[i])
        flo = readFlowFile(self.flo_paths[i])
        if self.transform is not None:
            img1,img2,flo = self.transform(img1,img2,flo)
        return img1, img2, flo

    def __len__(self):
        return len(self.img0_paths)
