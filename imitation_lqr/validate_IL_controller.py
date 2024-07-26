"""
Validate the IL controller and compare with the LQR controller
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from lqr_controller import lqr_controller
# from practice import NNController  # this is weird, call this will run practice!
import pickle as pkl
import torch.nn as nn

# cpu or gpu, that's a question
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the expert parameters
expert_file_path = 'work/expert.pkl'
with open(expert_file_path, 'rb') as f:
    expert = pkl.load(f)

A = expert['A'].to(device)
B = expert['B'].to(device)
Q = expert['Q'].to(device)
p = expert['p'].to(device)
T = expert['T']
u_lower = expert['u_lower']
u_upper = expert['u_upper']
time_steps = 20


n_state = A.size(0)
n_ctrl = B.size(1)
hidden_size = 32

# Define NN
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

# Initialize and load the NN controller
nn_controller = NNController(n_state, hidden_size, n_ctrl)

# Load the saved model state
model_path = 'model/model.ckpt'
nn_controller.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
nn_controller.eval()
nn_controller.to(device)

# Generate random initial points
n_points = 10
x_init = torch.rand(n_points, n_state).to(device)

# Generate trajectories
nn_trajectories = []
lqr_trajectories = []

for i in range(n_points):
    x_nn = x_init[i:i+1, :].clone()  # result in a tensor of shape (1, n_state)
    x_lqr = x_init[i:i+1, :].clone()  # result in a tensor of shape (1, n_state)

    nn_traj = [x_nn.cpu().numpy()]  # result in a numpy array
    lqr_traj = [x_lqr.cpu().numpy()]  # result in a numpy array

    # Simulate NN and LQR trajectories
    for t in range(time_steps):
        u_nn = nn_controller(x_nn)  # Compute the control input using NN controller
        _, u_seq = lqr_controller(x_init, A, B, Q, p, T, u_lower, u_upper)  # Compute the optimal control sequence using LQR controller
        u_lqr = u_seq[0:1, :]  # take the first control action and reshape into (1, n_ctrl)





