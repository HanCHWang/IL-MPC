"""
This code generates training and testing data for imitation_lqr.
"""
import torch
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
import os
from lqr_controller import lqr_controller

# Define parameters
n_batch, n_state, n_ctrl, T = 2, 3, 4, 10  # Example sizes
n_train, n_test = 2000, 1000  # Number of training and testing samples
n_sc = n_state + n_ctrl

# Set seeds for reproducibility
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Expert parameters
expert = {
    'Q': torch.eye(n_sc).to(device),
    'p': torch.randn(n_sc).to(device),
    'A': (torch.eye(n_state) + 0.2 * torch.randn(n_state, n_state)).to(device),
    'B': torch.randn(n_state, n_ctrl).to(device),
    'u_lower': None,
    'u_upper': None,
    'u_init': None
}

# Define data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

def generate_data(num_samples):
    x_data = []
    u_data = []

    for i in range(num_samples):
        x_init = 10-20*torch.randn(n_batch, n_state).to(device)

        F = torch.cat((expert['A'], expert['B']), dim=1) \
            .unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
        _, u_true, _ = lqr_controller(x_init, expert['A'], expert['B'], expert['Q'], expert['p'], expert['T'], None, None)

        x_data.append(x_init)
        u_data.append(u_true[0, :, :])  # Taking the control input at the first time step
        if (i + 1) % 10 == 0:
            print('Step [{}/{}]'.format(i + 1, num_samples, ))

    x_data = torch.cat(x_data, dim=0)
    u_data = torch.cat(u_data, dim=0)

    return x_data, u_data

# Generate training and test data
x_train, u_train = generate_data(n_train)
x_test, u_test = generate_data(n_test)

# Save the tensors in the specified directory
torch.save(x_train, os.path.join(data_dir, 'x_train.pt'))
torch.save(x_test, os.path.join(data_dir, 'x_test.pt'))
torch.save(u_train, os.path.join(data_dir, 'u_train.pt'))
torch.save(u_test, os.path.join(data_dir, 'u_test.pt'))
