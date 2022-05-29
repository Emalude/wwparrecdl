from __future__ import absolute_import, division, print_function

import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from scipy.stats import linregress
from ray import tune

a           = 270.
b           = 108.
d           = 0.154
gamma       = 0.641/1000
tau_s       = 100
tau_noise   = 2.
Jll = Jrr   = 0.2609
Jlr = Jrl   = 0.0497
J_ext       = 0.00052
I_o         = 0.3255
beta        = 1
sigma_noise = I_o / 16.275
mu_o        = 30
ndt         = 0
threshold   = 15
dt          = 1


def wangwong(coherence, Jii = 0.2609, Jij = 0.0497 beta = 1, ndt=0):
    """
    Run a single trial of the Wang & Wong (2006) model.

    Parameters
    ----------
    coherence: float - dot movement coherence. Must be between -1 and 1. Negative values for rightward movement.
    Jii: float  - self-excitatory coupling strength. Default: 0.2609 (Wang & Wong, 2006)
    Jij: float  - mutual-inhibitory coupling strength. Default: 0.0497 (Wang & Wong, 2006)
    beta: float - common background current modulation. Default: 1.
    ndt: int    - non-decision time in ms. Default: 0.
    
    Returns
    -------
    time, choice: tuple - reaction time between 0 and 2500ms and choice: 0 if wrong, 1 if correct, -1 if not decided
    """

    # Input from stimulus
    I_mot_l = J_ext * mu_o * (1 + coherence *1. / 100)
    I_mot_r = J_ext * mu_o * (1 - coherence *1. / 100)

    sl   = random.random() * 0.1
    sr   = random.random() * 0.1
    I_n1 = random.random() * 0.1
    I_n2 = random.random() * 0.1
   

    for i in range(2500):

        I_l = Jii * sl - Jij*sr + I_mot_l + beta*I_o + I_n1
        I_r = Jii * sr - Jij*sl + I_mot_r + beta*I_o + I_n2

        r_l = (a*I_l - b)/(1 - np.exp(-d*(a*I_l - b)))
        r_r = (a*I_r - b)/(1 - np.exp(-d*(a*I_r - b)))

        sl += (-sl*1./ tau_s + (1 - sl) * gamma * r_l)*dt
        sr += (-sr*1./ tau_s + (1 - sr) * gamma * r_r)*dt

        I_n1 += (- I_n1 + random.random() * math.sqrt(tau_noise) * sigma_noise)*dt / tau_noise
        I_n2 += (- I_n2 + random.random() * math.sqrt(tau_noise) * sigma_noise)*dt / tau_noise

        sl_hz = sl *1./ ((1 - sl) * gamma * tau_s)
        sr_hz = sr *1./ ((1 - sr) * gamma * tau_s)

        if threshold - sl_hz < 1e-9 or threshold - sr_hz < 1e-9:
            time   = i + ndt
            choice = sl_hz > sr_hz
            break

        time=i
        choice=-1

    return choice, time

def accuracy(outcomes):
    """
    Compute accuracy.
    
    Parameters
    ----------
    outcomes: array-like - list of tuples or list of list number of outcomes x (choice, reaction time).
    
    Returns
    -------
    Accuracy: float - fraction of correct choices over decisions made (non-decisions not counted).
              If there are no decisions, returns -1.
    """
    correct = np.array([choice[0] == 1 for choice in outcomes])
    incorrect = np.array([choice[0] == 0 for choice in outcomes])
    decided = np.sum(correct) + np.sum(incorrect)
    if decided != 0:
        return np.sum(correct)/decided
    else:
        return -1

def reaction_times(outcomes, agg_func = np.mean, correct = True):
    """
    Compute statistics of reaction time. Default: average reaction time.

    Parameters
    ----------
    outcomes: array-like - list of tuples or list of list number of outcomes x (choice, reaction time).
    agg_func: function - statistic to compute for reaction times. Default: np.mean
    correct: bool - statistic to be calculated for correct choices. Default: True.
    
    Returns
    -------
    statistic: statistic defined by agg_func applied to the reaction times.
    """
    return agg_func([outcome[1] for outcome in outcomes if outcome[0] == correct])

class PRNet(nn.Module):
    def __init__(self, input_size, l1, l2, l3, output_size):
        super(PRNet, self).__init__()
        self.input_size = input_size
        self.fc1   = nn.Linear(self.input_size, l1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(l1, l2)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(l2, l3)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(l3, output_size)

    def forward(self, x):
        output = self.relu1(self.fc1(x))
        output = self.relu2(self.fc2(output))
        output = self.relu3(self.fc3(output))
        output = self.output(output)
        return output

def load_data(data_dir, parameter, coherence=all):
    
    features = np.load(str(data_dir / f'features_{parameter}_{coherence}.npy'), allow_pickle = True)
    parameters = np.load(str(data_dir / f'parameters_{parameter}_{coherence}.npy'), allow_pickle = True)
    n_pars = parameters.shape[1]    

    features = features[:,:,:]
    valid = np.multiply(features[:,:,0] < .999, features[:,:,1] < 0.2)
    cond = [all(valid[i]) for i in range(len(valid))]
    raw_data = np.concatenate((np.expand_dims(features[cond][:,:,0], 2), features[cond][:,:,2:]), 2).astype(float)
    parameters = parameters[cond]
    data = np.reshape(raw_data,(raw_data.shape[0], -1))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = torch.Tensor(data).to(device).float()
    y = torch.Tensor(parameters).to(device).float()

    train_size = int(0.8 * len(data))

    trainset = TensorDataset(X[:train_size], y[:train_size])
    testset  = TensorDataset(X[train_size:], y[train_size:])

    return trainset, testset

def train(model, loss_function, optimizer, train_data, test_data, num_epochs=25):
    for epoch in range(0, num_epochs):

        print(f'Starting epoch {epoch+1}')

        current_loss = 0.0
        for i, (inputs, targets) in enumerate(train_data, 0):                 
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = loss_function(outputs, targets)
            
            train_loss.backward()
            optimizer.step()

            current_loss += train_loss.item()
           
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data[:][0])
            test_loss = loss_function(test_outputs, test_data[:][1])
            
        model.train()
        print(f'Epoch {epoch}:\n'
              f'Train loss: {current_loss:.4f}\n'
              f'Test loss: {test_loss:.4f}\n'
              f'----------')

    print('Training process has finished. Saving the model...')

    return model.state_dict()

def train_tune(config, checkpoint_dir = None, data_dir=None):

    train_set, test_set = load_data(data_dir, "3_pars", "all")
    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(train_set, [test_abs, len(train_set) - test_abs])
    in_dim = train_set[0][0].shape[0]
    out_dim   = train_set[0][1].shape[0]
    
    model = PRNet(in_dim, config['l1'], config['l2'], config['l3'], out_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    model.to(device)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], momentum = config['mom'])
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(checkpoint_dir / 'checkpoint')
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader = DataLoader(train_subset, batch_size=int(config['batch_size']), shuffle=True)
    valloader   = DataLoader(val_subset, batch_size=int(config['batch_size']), shuffle=True)

    for epoch in range(50):
   
        epoch_steps = 0
        current_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)      
                 
            optimizer.zero_grad()

            outputs = model(inputs)
            train_loss = loss_function(outputs, targets)
            train_loss.backward()
            optimizer.step()

            current_loss += train_loss.item()
            epoch_steps += 1
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                current_loss / epoch_steps))
                current_loss = 0.0

        model.eval()
        R = np.zeros(out_dim)
        val_loss = 0.0
        val_steps = 0
        for i, (inputs, targets) in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, targets = inputs.to(device), targets.to(device)
                test_outputs = model(inputs)
                loss = loss_function(test_outputs, targets)
                for r in range(len(R)):
                    try:
                        linreg = linregress(test_outputs.cpu().numpy()[:,r], targets.cpu().numpy()[:,r])
                        R[r] += linreg.rvalue
                    except ValueError:
                        continue
                val_loss += loss.detach().cpu().numpy()
                val_steps += 1
        model.train()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = checkpoint_dir + "checkpoint"
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        if val_steps != 0:
            tune.report(loss = val_loss/val_steps,
                        R2_1 = (R[0]/val_steps)**2,
                        R2_2 = (R[1]/val_steps)**2,
                        R2_3 = (R[2]/val_steps)**2
                        )
        else:
            tune.report(loss=np.nan, R2_1=np.nan, R2_2=np.nan, R2_3=np.nan)

    print("Finished Training")
    


    
