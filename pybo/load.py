import torch
import numpy as np
import csv
import pandas as pd
from tc_learn import *

# model = Net()

# checkpoint = torch.load('./DNN_model.pt')
# model.load_state_dict(checkpoint['model'])

# str = torch.Tensor([-0.332102407,2.325789922,0.620879471,0.336402609,-0.042428026,-0.036377326,-0.033060542])
# result = model(str)
# print(result)
 
str = torch.Tensor([-0.332102407,2.325789922,0.620879471,0.336402609,-0.042428026,-0.036377326,-0.033060542])
str.unsqueeze_(0)
W= torch.Tensor([[1.1432, 0.9258, 0.8426, 0.8354, 0.8311],
        [0.9211, 0.9496, 0.9636, 1.0065, 1.1431],
        [1.0667, 1.6569, 0.9356, 0.4278, 0.3309],
        [0.9891, 1.0194, 1.0133, 0.9847, 0.9710],
        [1.8983, 0.1123, 0.1053, 1.8552, 0.3555],
        [1.8921, 1.8494, 0.1108, 0.1072, 0.1060],
        [1.8987, 0.1117, 0.1043, 0.9887, 0.0963]])
b= torch.Tensor([0.3324])
result = F.softmax(str.matmul(W) + b, dim=1)
maxv = 0.0
for line in result[0]:
    if line.item() > maxv:
        maxv = line.item()