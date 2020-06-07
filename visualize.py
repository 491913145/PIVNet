from models.PWCNet import pwc_dc_net
import torch
from torch.autograd import Variable
from torchviz import make_dot

model = pwc_dc_net()
model.cuda()

from torchviz import make_dot
x = Variable(torch.randn(2,1, 1,256,256).cuda())
vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
vis_graph.view()