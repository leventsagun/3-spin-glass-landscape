# related research https://github.com/facebookresearch/DeepConvexity
# thanks DLP for pointing at einsum that helped increase efficiency a lot!

import os
import time
import math
import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def run_experiment(args):
    
    def energy(x):
        # feed already normalized
        noise = torch.einsum("ijk,i,j,k", (coeff, x, x, x)) / d
        signal = (x[0])**3 / math.sqrt(d)
        return noise - args.coeff * signal
        
    d = args.dimension
    
    # pick couplings
    torch.manual_seed(args.seed_J)
    t = time.time()
    coeff = torch.randn(d, d, d).to(args.device)
    print('time to sample {:.3f}'.format(t - time.time()))

    # pick initial state
    torch.manual_seed(args.seed_s)
    x = torch.randn(d).to(args.device)
    x.requires_grad_()
    x.data /= x.data.norm() / math.sqrt(d)
    
    if args.initial_height != 0:
        # set the first coordinate
        x_0 = torch.tensor(args.initial_height).mul(math.sqrt(d))
        assert 0 < x_0 <= math.sqrt(d)
        x.data[0].copy_(x_0)
        x.data[1:] /= x.data[1:].norm() / math.sqrt(d - x_0 ** 2)

    # optimizer and lr scheduling
    optimizer = torch.optim.SGD([x], lr=args.lr, momentum=args.mom)
    if args.schedule:
        scheduler = ReduceLROnPlateau(
                optimizer,
                patience=2, # num iter to wait
                factor=0.99,
                threshold=1e-7,
                min_lr=1e-7,
                threshold_mode='abs',
                erbose=False)
    
    # record the values at initialization
    val = energy(x)
    val.backward()

    p_grad = torch.dot(x.grad.data, x.data.div(x.norm())) * x.data.div(x.norm())
    val_grad_norm = (x.grad.data - p_grad).norm().item()

    history = [[
        optimizer.param_groups[0]['lr'],
        (x[0] / math.sqrt(d)).item(), # init overlap
        val.item(), # init loss
        val_grad_norm
        ]]
    
    for i in range(args.n_iters):

        optimizer.step()
        optimizer.zero_grad()
        if args.schedule:
            scheduler.step(val.item())
        x.data /= x.data.norm() / math.sqrt(d)

        val = energy(x)
        val.backward()

        # project the gradient to the sphere for the calculation of its norm
        p_grad = torch.dot(x.grad.data, x.data.div(x.norm())) * x.data.div(x.norm())
        val_grad_norm = (x.grad.data - p_grad).norm().item()

        history.append([
            optimizer.param_groups[0]['lr'],
            (x[0] / math.sqrt(d)).item(),
            val.item(), 
            val_grad_norm
            ])

        # early stopping condition
        if val_grad_norm < args.eps:
            print('early stopping at {}'.format(i))
            break
        
    return torch.tensor(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='spiked tensor model')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--save_file', type=str, default='result.hist')
    parser.add_argument('--dimension', type=int, default=400)
    parser.add_argument('--coeff', type=float, default=10)
    parser.add_argument('--n_iters', type=int, default=3500)
    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mom', type=float, default=0)
    parser.add_argument('--seed_s', type=int, default=0)
    parser.add_argument('--seed_J', type=int, default=0)
    parser.add_argument('--initial_height', type=float, default=0)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--no_save', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=False)
    args = parser.parse_args()
    
    # initial setup
    if args.double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if use_cuda else 'cpu')
    
    print(args)
    
    file_path = os.path.join(args.save_dir, args.save_file)
    assert os.path.exists(args.save_dir)
    assert not os.path.exists(file_path)

    t = time.time()
    history = run_experiment(args)
    print('time per run {:.3f}'.format(t - time.time()))
    torch.save(history, file_path)

