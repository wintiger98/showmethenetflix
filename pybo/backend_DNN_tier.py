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

params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

num = np.arange(5416)
np.random.permutation(num)

temp = movieq[num[0]:num[3613]]
t_un_train = temp[:,:-5]
t_c_train = temp[:,-1]

temp = movieq[num[3613]:num[5415]]
t_un_val = temp[:,:-5]
t_c_val = temp[:,-1]

linear_model = Net()

optimizer = optim.Adam(
    linear_model.parameters(), # <2>
    lr=1e-2)

training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = linear_model,
    loss_fn = loss_fn,
    t_u_train = t_un_train,
    t_u_val = t_un_val, 
    t_c_train = t_c_train,
    t_c_val = t_c_val)

torch.save({
            'model' : linear_model.state_dict()
}, './model/DNN_model.pt')

