"""
This code generates training and testing data for imitation_lqr.
"""
import torch
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
import os
from lqr_controller import lqr_controller
import pickle as pkl

# Define parameters
n_batch, n_state, n_ctrl = 2, 3, 4  # Example sizes
n_train, n_test = 2000, 1000  # Number of training and testing samples
n_sc = n_state + n_ctrl

# Set seeds for reproducibility
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define expert parameters
expert = {
    'Q': torch.eye(n_sc),
    'p': torch.randn(n_sc),
    'A': (torch.eye(n_state) + 0.2 * torch.randn(n_state, n_state)),
    'B': torch.randn(n_state, n_ctrl),
    'u_lower': None,
    'u_upper': None,
    'u_init': None,
    'T': 10
}

# Save expert parameters
fname = os.path.join('work', 'expert.pkl')
with open(fname, 'wb') as f:
    pkl.dump(expert, f)

# Define data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

A = expert['A'].to(device)
B = expert['B'].to(device)
Q = expert['Q'].to(device)
p = expert['p'].to(device)
T = expert['T']
def generate_data(num_samples):
    x_data = []
    u_data = []

    for i in range(num_samples):
        x_init = 10-20*torch.randn(n_batch, n_state).to(device)

        F = torch.cat((A, B), dim=1) \
            .unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
        _, u_true = lqr_controller(x_init, A, B, Q, p, T, None, None)

        x_data.append(x_init)

        # Taking the control input at the first time step, and reshape
        # into (n_batch, n_ctrl)
        u_data.append(u_true[0, :, :])
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
