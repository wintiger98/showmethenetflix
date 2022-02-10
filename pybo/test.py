import torch
import numpy as np
import csv
import pandas as pd
from tc_learn import *

torch.set_printoptions(edgeitems = 2)

movie_path = "./data/bit_data_zscore_tier.csv"
movieq_numpy = pd.read_csv(movie_path, delimiter=",", encoding= 'cp949')
col_list = next(csv.reader(open(movie_path), delimiter=','))
mov = movieq_numpy[['genre','rating','country','score','company','writer','director','hit rate','zscore','robust','minmax','tier']]
movieq = torch.from_numpy(mov.values).float()
torch.manual_seed(1)

target_one_hot = torch.zeros(5416, 5, dtype=torch.float)
target = movieq[:,-1]

for i in range(5416):
    target_one_hot[i][int(target[i])-1] = 1

W= torch.Tensor([[1.1432, 0.9258, 0.8426, 0.8354, 0.8311],
        [0.9211, 0.9496, 0.9636, 1.0065, 1.1431],
        [1.0667, 1.6569, 0.9356, 0.4278, 0.3309],
        [0.9891, 1.0194, 1.0133, 0.9847, 0.9710],
        [1.8983, 0.1123, 0.1053, 1.8552, 0.3555],
        [1.8921, 1.8494, 0.1108, 0.1072, 0.1060],
        [1.8987, 0.1117, 0.1043, 0.9887, 0.0963]])
b= torch.Tensor([0.3324])
data = movieq[:,:-5]
result = F.softmax(data.matmul(W) + b, dim=1)

for i in range(5416):
    maxv = 0.0
    maxindex = -1
    for j in range(5):
        if result[i][j].item() > maxv:
            maxindex = j
            maxv = result[i][j].item()
    
    print(data[i],maxindex+1,target[i])