import sys

parameter = sys.argv[1]
coherence = sys.argv[2]

import wwparrecdl as pr

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import numpy as np

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_path = Path(__file__).resolve().parent.parent.absolute() / 'wwparrecdl/data'
model_dir = data_path / 'models'
data_dir = data_path / 'dl_input' 

train_set, test_set = pr.load_data(data_dir, parameter = parameter, coherence=coherence)

# ----- Set hyperparameters here -----
batch_size = 50
n_features = 13
n_points   = 1
layer_1_neurons = 128
layer_2_neurons = 128
layer_3_neurons = 256
n_outputs  = 3

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

prnet = pr.PRNet(n_features*n_points, layer_1_neurons, layer_2_neurons, layer_3_neurons, n_outputs)

if torch.cuda.device_count() > 1:
    prnet = DataParallel(net)

prnet.to(device)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(prnet.parameters(), lr= 0.000496637, momentum = 0.795967)

state_dict = pr.train(prnet, loss_function, optimizer, train_dataloader, test_set, num_epochs=50)

pars = '3_pars'
torch.save(state_dict, str(model_dir / pars / f'PRNet_{coherence}'))



