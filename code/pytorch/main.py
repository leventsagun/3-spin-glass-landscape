import argparse
import torch
import time
import math
import os


def run_experiment(args, d, lr):

    # set the coefficients
    coeff = torch.randn(d, d, d)

    def energy(x):
        x = x / x.norm() * math.sqrt(d)
        tmp1 = torch.mul(x.view(d, 1), x.view(1, d))
        tmp2 = torch.mul(tmp1.view(d * d, 1), x.view(1, d))
        tmp3 = torch.mul(tmp2.view(d * d * d), coeff.view(1, -1))
        return tmp3.sum() / d

    # initial point
    x = torch.randn(d, requires_grad=True) 
    x.data = x / x.norm() * math.sqrt(d)

    loss = energy(x)
    loss.backward()

    # initial value of the Hamiltonian and the norm of its gradient
    print(loss.item()) 

    for i in range(args.n_iters):
        x.data = x - lr * x.grad
        x.grad.data.zero_
        
        loss = energy(x)
        loss.backward()
        print(loss.item()) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3-spin model')
    parser.add_argument('--save', type=str, default='results/')
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--n_iters', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    torch.set_default_tensor_type(torch.DoubleTensor)

    run_experiment(args, args.dim, args.lr)
    

