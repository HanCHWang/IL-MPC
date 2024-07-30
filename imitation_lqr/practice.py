#!/usr/bin/env python3

import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import torch.optim as optim

import numpy as np
import numpy.random as npr

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost, LinDx

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import time
import os
import shutil
import pickle as pkl
import collections

import argparse
import setproctitle

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data directory
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# Load tensors from the specified directory
x_train = torch.load(os.path.join(data_dir, 'x_train.pt'))
x_test = torch.load(os.path.join(data_dir, 'x_test.pt'))
u_train = torch.load(os.path.join(data_dir, 'u_train.pt'))
u_test = torch.load(os.path.join(data_dir, 'u_test.pt'))

# Create TensorDatasets
train_dataset = torch.utils.data.TensorDataset(x_train, u_train)
test_dataset = torch.utils.data.TensorDataset(x_test, u_test)

# Create DataLoaders
batch_size = 5
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Load the expert parameters
expert_file_path = 'work/expert.pkl'
with open(expert_file_path, 'rb') as f:
    expert = pkl.load(f)

# Hyper parameters
# parser = argparse.ArgumentParser()
# parser.add_argument('--n_state', type=int, default=3)
# parser.add_argument('--n_ctrl', type=int, default=4)
# parser.add_argument('--T', type=int, default=5)
# parser.add_argument('--save', type=str)
# parser.add_argument('--work', type=str, default='work')
# parser.add_argument('--no-cuda', action='store_true')
# parser.add_argument('--seed', type=int, default=0)
# args = parser.parse_args()

n_batch = 64
num_layers = 2
hidden_size = 32
learning_rate = 1e-4
num_epochs = 500

# args.cuda = not args.no_cuda and torch.cuda.is_available()
# t = '.'.join(["{}={}".format(x, getattr(args, x))
#                 for x in ['n_state', 'n_ctrl', 'T']])
# setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
# if args.save is None:
#     args.save = os.path.join(args.work, t, str(args.seed))

# if os.path.exists(args.save):
#     shutil.rmtree(args.save)
# os.makedirs(args.save, exist_ok=True)

# expert_seed = 42
# assert expert_seed != args.seed
# torch.manual_seed(expert_seed)
#
# Q = torch.eye(n_sc)
# p = torch.randn(n_sc)

# alpha = 0.2  # magnitude for the state matrix A

Q = expert['Q'].to(device)
p = expert['p'].to(device)
A = expert['A'].to(device)
B = expert['B'].to(device)
u_lower = expert['u_lower']
u_upper = expert['u_upper']
delta = None
u_init = None
n_state, n_ctrl, T = A.size(0), B.size(1), expert['T']
n_sc = n_state + n_ctrl


# Define the NN controller
class NNController(nn.Module):
    def __init__(self, n_state, hidden_size, n_ctrl):
        super(NNController, self).__init__()
        self.hidden_size = hidden_size

        # Initialize weights and biases for all layers
        self.fc1 = nn.Linear(n_state, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, n_ctrl)

    def forward(self, x):  # x: (n_batch, n_state)
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.fc3(out)
        return out


# Define the loss function
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.norm(predictions-targets)
        return loss


# Construct the NN model
model = NNController(n_state, hidden_size, n_ctrl)

# Loss and optimizer
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# store the loss values
loss_values = []

# Train the model!
total_step = len(train_loader)
for epoch in range(num_epochs):  # episode size
    for i, (x_train, u_train) in enumerate(train_loader):
        # Move tensors to the configured device
        x_train = x_train.to(device)
        u_train = u_train.to(device)

        # Forward pass
        predictions = model(x_train)
        loss = criterion(predictions, u_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store the loss value
        loss_values.append(loss.item())

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i + 1, total_step, loss.item()))


# Test the model
with torch.no_grad():
    loss_value = []
    for x_test, u_test in test_loader:
        x_test = x_test.to(device)
        u_test = u_test.to(device)
        predictions = model(x_test)
        loss_value.append(criterion(predictions, u_test))

    print('Test Loss: {:.4f}'.format(np.mean(loss_value)))

# Save the model checkpoint
data_dir = os.path.join(os.path.dirname(__file__), 'model')
torch.save(model.state_dict(), os.path.join(data_dir, 'model.ckpt'))

