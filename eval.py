from torchsummary import summary
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
import os
from models.PWCNet import pwc_dc_net
import torch
from torch.autograd import Variable
from utils.multiscaleloss import realEPE
from tqdm import tqdm
from utils.dataloader import MyDataset



def load_data(path):
    flo_paths = glob(os.path.join(path, '*.flo'))
    img0_paths = [x.replace('flow.flo', 'img1.tif') for x in flo_paths]
    img1_paths = [x.replace('flow.flo', 'img2.tif') for x in flo_paths]
    return img0_paths, img1_paths, flo_paths


def find_NewFile(path):
    # 获取文件夹中的所有文件
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路径
    file_new = lists[-1]
    return file_new


weight_path = find_NewFile('logname')
# weight_path = 'pwc_net.pth.tar'
model = pwc_dc_net(weight_path)
# model = nn.DataParallel(model) #就单GPU走一波吧
model.cuda()
model.eval()
summary(model, input_size=(1, 1, 256, 256))

print('load weight: ',weight_path)

dataset = MyDataset('/home/disk/lihaiyun/LiteFlow/PIV-LiteFlowNet-en/lite/dataset/splits/test', shape=(256, 256))
ImgLoader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=16,shuffle=True, pin_memory=True)
total_test_aepe = []

iterator = iter(ImgLoader)
step = len(ImgLoader)

for i in tqdm(range(step)):
    img1, img2, flo = next(iterator)
    img1 /= 255
    img2 /= 255
    img1 = Variable(torch.FloatTensor(img1.float()))
    img2 = Variable(torch.FloatTensor(img2.float()))
    flo = Variable(torch.FloatTensor(flo.float()))
    imgL, imgR, flowl0 = img1.cuda(), img2.cuda(), flo.cuda()
    output = model((imgL, imgR))
    total_test_aepe += [realEPE(output.detach(), flowl0.detach()).cpu().numpy()]

print("aepe: ",np.mean(total_test_aepe))
