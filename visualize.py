from models.PWCNet import pwc_dc_net
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils.dataloader import MyDataset
from utils.augmentations import Basetransform
from utils.flowlib import *
import torch.nn.functional as F
import os
from glob import glob
import time
import tensorwatch as tw

def find_NewFile(path):
    # 获取文件夹中的所有文�?
    lists = glob(os.path.join(path, '*.tar'))
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(x))
    # 把目录和文件名合成一个路�?
    file_new = lists[-1]
    return file_new

model = pwc_dc_net()
model.cuda()
model.eval()

from torchviz import make_dot
# x = Variable(torch.randn(2,1, 3,256,256).cuda())
# vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
# vis_graph.view()
tw.draw_model(model, [2,1, 3, 224, 224])

def main():
    weight_path = find_NewFile('logname')
    # weight_path = 'logname/finetune_60.tar'
    model = pwc_dc_net(weight_path)
    model.cuda()
    model.eval()
    x1 = Variable(torch.zeros((1, 3, 128, 128))).cuda()
    x2 = Variable(torch.zeros((1, 3, 128, 128))).cuda()
    for i in range(10):
        t_s = time.time()
        model((x1,x2))
        t_d = time.time()
        print((t_d-t_s)*1000)
    # summary(model, input_size=(1, 1, 256, 256))

    # print('load weight: ', weight_path)

    # dataset = MyDataset('/home/disk/lihaiyun/LiteFlow/PIV-LiteFlowNet-en/lite/dataset/splits/test',transform=Basetransform())
    # ImgLoader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)
    # # rmse = eval(model,ImgLoader)
    # step = len(ImgLoader)
    # for i in tqdm(range(step)):
    #     iterator = iter(ImgLoader)
    #     img1,img2,flo_gt = next(iterator)
    #     img1 = Variable(torch.FloatTensor(img1.float()))
    #     img2 = Variable(torch.FloatTensor(img2.float()))
    #     imgL, imgR = img1.cuda(), img2.cuda()
    #     flo = model((imgL, imgR)).detach()
    #     flo = F.interpolate(flo, (256, 256), mode='bilinear', align_corners=False).cpu().numpy()[0]
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plotFlow_Li(flo.transpose(1,2,0))
    #     plt.subplot(1, 2, 2)
    #     plotFlow_Li(flo_gt.numpy()[0].transpose(1,2,0))
    #     plt.show()
# main()