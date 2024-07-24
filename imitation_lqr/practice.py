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

# Define data directory
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

# Create TensorDatasets
x_train = torch.load(os.path.join(data_dir, 'x_train.pt'))
x_test = torch.load(os.path.join(data_dir, 'x_test.pt'))
u_train = torch.load(os.path.join(data_dir, 'u_train.pt'))
u_test = torch.load(os.path.join(data_dir, 'u_test.pt'))

# Hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_state', type=int, default=3)
parser.add_argument('--n_ctrl', type=int, default=3)
parser.add_argument('--T', type=int, default=5)
parser.add_argument('--save', type=str)
parser.add_argument('--work', type=str, default='work')
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

n_batch = 128
num_layers = 2
hidden_size = 128
learning_rate = 1e-3

args.cuda = not args.no_cuda and torch.cuda.is_available()
t = '.'.join(["{}={}".format(x, getattr(args, x))
                for x in ['n_state', 'n_ctrl', 'T']])
setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
if args.save is None:
    args.save = os.path.join(args.work, t, str(args.seed))

if os.path.exists(args.save):
    shutil.rmtree(args.save)
os.makedirs(args.save, exist_ok=True)

device = 'cuda' if args.cuda else 'cpu'

n_state, n_ctrl, T = args.n_state, args.n_ctrl, args.T
n_sc = n_state+n_ctrl

expert_seed = 42
assert expert_seed != args.seed
torch.manual_seed(expert_seed)

Q = torch.eye(n_sc)
p = torch.randn(n_sc)

alpha = 0.2

expert = dict(
    Q = torch.eye(n_sc).to(device),
    p = torch.randn(n_sc).to(device),
    A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state)).to(device),
    B = torch.randn(n_state, n_ctrl).to(device),
    u_lower = None,
    u_upper = None,
    delta = None,
    u_init = None
)

fname = os.path.join(args.save, 'expert.pkl')
with open(fname, 'wb') as f:
    pkl.dump(expert, f)

torch.manual_seed(args.seed)
A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state))\
    .to(device).requires_grad_()
B = torch.randn(n_state, n_ctrl).to(device).requires_grad_()


# Define the NN controller
class NNController(nn.Module):
    def __init__(self, n_state, hidden_size, num_layers, n_ctrl):
        super(NNController, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
model = NNController(n_state, hidden_size, num_layers, n_ctrl)

# Loss and optimizer
criterion = get_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the state-input tensor
F = torch.cat((A, B), dim=1) \
                .unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)

# store the loss values
loss_values = []

# Train the model!
for i in range(500):  # episode size
    x_init = torch.rand(n_batch, n_state).to(device)

    # Get the targets
    x_true, u_true, objs_true = mpc.MPC(
        n_state, n_ctrl, T,
        u_lower=expert['u_lower'], u_upper=expert['u_upper'], u_init=expert['u_init'],
        lqr_iter=100,
        verbose=-1,
        exit_unconverged=False,
        detach_unconverged=False,
        n_batch=n_batch,
    )(x_init, QuadCost(expert['Q'], expert['p']), LinDx(F))

    # Forward pass
    predictions = model(x_init)
    targets = u_true[0, :, :]
    loss = criterion(predictions, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # store the loss value
    loss_values.append(loss.item())

    if (i + 1) % 10 == 0:
        print('Step [{}/{}], Loss: {:.4f}'.format(i + 1, 500, loss.item()))


