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

W = torch.ones((7, 5), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.Adam([W, b], lr=0.00001)
data = movieq[:,:-5]
epochs = 90000
for epoch in range(epochs+1):

    # 가설
    hypothesis = F.softmax(data.matmul(W) + b, dim=1) 

    # 비용 함수
    cost = (target_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

        # 100번마다 로그 출력
    if epoch % 10000 == 0:
        print('Epoch {:4d} Cost: {:.6f}'.format(epoch, cost.item()))

print(W,b)
