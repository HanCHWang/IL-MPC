"""
This code designs lqr_controller() to generate the open-loop optimal trajectory for given
system and initial conditions.

x_init: initial condition vector, stored as a tensor of shape (n_batch, n_state)
A: state matrix
B: input matrix
Q: matrix for the quadratic cost
p: vector for the linear cost
T: prediction horizon
u_lower, u_upper: lower and upper bounds for the control input
"""
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
import torch

def lqr_controller(x_init, A, B, Q, p, T, u_lower, u_upper):
    n_state = A.size(0)  # dimension of the state
    n_ctrl = B.size(0)  # dimension of the control
    n_batch = x_init.size(0)  # batch size

    F = torch.cat((A, B), dim = 1).unsqueeze(0).unsqueeze(0).repeat(T, n_batch, 1, 1)
    x, u, _ = mpc.MPC(
        n_state, n_ctrl, T,
        u_lower=u_lower, u_upper=u_upper,
        lqr_iter=100,
        verbose=-1,
        exit_unconverged=False,
        detach_unconverged=False,
        n_batch=n_batch,
    )(x_init, QuadCost(Q,p), LinDx(F))

    return x, u