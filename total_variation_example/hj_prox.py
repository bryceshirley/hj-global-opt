# MIT License
# Copyright (c) 2022 CSM Optimization and Machine Learning group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from operators import Finite_Difference_Gradient_2D

from scipy.special import roots_hermite

# Define the TV norm function
def tv_norm(x):
    """
    Computes the Total Variation (TV) norm of the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (int_samples * RGB_dim * pixels * pixels, 1) 
                          or (RGB_dim * pixels * pixels, 1).

    Returns:
        torch.Tensor: TV norm for each sample, shape (int_samples, 1).
    """

    # Reshape the input tensor
    x = x.view(-1, 3, 512, 512)
    # Initialize the Finite Difference Gradient operator
    G = Finite_Difference_Gradient_2D()

    # Compute the Finite Difference Gradient of x
    result = G(x)
    
    # Compute the TV norm and sum across spatial dimensions and channels
    tv_norm = torch.norm(result, p=1, dim=(1, 2, 3), keepdim=True) # shape (int_samples, 1)
    tv_norm = tv_norm.squeeze() # shape (int_samples)
    return tv_norm

def h(tau, direction, xk,int_samples,dims):
    

    xk_expanded = xk.expand(int_samples, dims)
    direction_expanded = direction.expand(int_samples, dims)

    y = xk_expanded + tau * direction_expanded
    
    return tv_norm(y)

def HJ_prox(x, f=tv_norm, t=1e-1, delta=1e-1, int_samples=100, alpha=1.0, linesearch_iters=0, device='cpu'):
    """ Estimate proximals from function value sampling via HJ-Prox Algorithm.

        The output estimates the proximal:
        
        $$
            \mathsf{prox_{tf}(x) = argmin_y \ f(y) + \dfrac{1}{2t} \| y - x \|^2,}
        $$
            
        where $\mathsf{x}$ = `x` is the input, $\mathsf{t}$=`t` is the time parameter, 
        and $\mathsf{f}$=`f` is the function of interest. The process for this is 
        as follows.
        
        - [x] Sample points $\mathsf{y^i}$ (via a Gaussian) about the input $\mathsf{x}$
        - [x] Evaluate function $\mathsf{f}$ at each point $\mathsf{y^i}$
        - [x] Estimate proximal by using softmax to combine the values for $\mathsf{f(y^i)}$ and $\mathsf{y^i}$            

        Note: 
            The computation for the proximal involves the exponential of a potentially
            large negative number, which can result in underflow in floating point
            arithmetic that renders a grossly inaccurate proximal calculation. To avoid
            this, the "large negative number" is reduced in size by using a smaller
            value of alpha, returning a result once the underflow is not considered
            significant (as defined by the tolerances "tol" and "tol_underflow").
            Utilizing a scaling trick with proximals, this is mitigated by using
            recursive function calls.
            
        Warning:
            Memory errors can occur if too many layers of recursion are used,
            which can happen with tiny delta and large f(x). 

        Args:
            x (tensor): Input vector
            t (tensor): Time > 0
            f (Callable): Function to minimize
            delta (float, optional): Smoothing parameter
            int_samples (int, optional): Number of samples in Monte Carlo sampling for integral
            alpha (float, optional): Scaling parameter for sampling variance
            linesearch_iters (int, optional): Number of steps used in recursion (used for numerical stability)
            device (string, optional): Device on which to store variables

        Shape:
            - Input `x` is of size `(n, 1)` where `n` is the dimension of the space of interest
            - The output `prox_term` also has size `(n, 1)`

        Returns:
            prox_term (tensor): Estimate of the proximal of f at x
            linesearch_iters (int): Number of steps used in recursion (used for numerical stability)
            envelope (tensor): Value of envelope function (i.e. infimal convolution) at proximal
            
        Example:
            Below is an exmaple for estimating the proximal of the L1 norm. Note the function
            must have inputs of size `(n_samples, n)`.
            ```
                def f(x):
                    return torch.norm(x, dim=1, p=1) 
                n = 3
                x = torch.randn(n, 1)
                t = 0.1
                prox_term, _, _ = compute_prox(x, t, f, delta=1e-1, int_samples=100)   
            ```
    """
    assert x.shape[1] == 1
    assert x.shape[0] >= 1

    dim = x.shape[0]

    
    z, weights = roots_hermite(int_samples)
    z_line = torch.tensor(z_line, dtype=torch.double)
    weights = torch.tensor(weights, dtype=torch.double)

    # Reshape xk to ensure broadcasting
# Ensure xk has the correct shape for broadcasting (int_samples, n_features)
    xk_squeezed = x.squeeze(0)  # Remove unnecessary dimensions (if xk has shape [1, 1, n_features])
    xk_expanded = xk_squeezed.expand(int_samples, dim)  # Shape: (int_samples_line, n_features)

    # Reshape z_line to allow broadcasting, then expand
    z_expanded = z.unsqueeze(1)  # Shape: (int_samples, 1)
    z_expanded = z_expanded.expand(int_samples, dim)  # Shape: (int_samples_line, n_features)

    
    while True:
        # Apply Rescaling to time
        t_rescaled = t/rescale_factor

        sigma = np.sqrt(2*delta*t_rescaled)

        # Compute Perturbed Points
        y = xk_expanded - (sigma * z_expanded)


    
    # linesearch_iters +=1
    # standard_dev = np.sqrt(delta * t / alpha)
    # dim = x.shape[0]
    
    # y = standard_dev * torch.randn(int_samples, dim, device=device) + x.permute(1,0) # y has shape (n_samples, dim)
    # z = -f(y)*(alpha/delta)     # shape =  n_samples
    # w = torch.softmax(z, dim=0) # shape = n_samples 
    
    # softmax_overflow = 1.0 - (w < np.inf).prod()
    # if softmax_overflow:
    #     alpha *= 0.5
    #     return HJ_prox(x, t=t, f=f, delta=delta, int_samples=int_samples, alpha=alpha,
    #                         linesearch_iters=linesearch_iters, device=device)
    # else:
    #     prox_term = torch.matmul(w.t(), y) # w.t() is shape (1,n_samples), y is shape (n_samples, dim) so result is shape (1, dim)
    #     prox_term = prox_term.view(-1,1) # shape = (dim, 1)
    
    # prox_overflow = 1.0 - (prox_term < np.inf).prod()
    # assert not prox_overflow, "Prox Overflowed"

    # #envelope = f(prox_term.view(1,-1)) + (1/(2*t)) * torch.norm(prox_term - x.permute(1,0), p=2)**2    
    # return prox_term #, linesearch_iters#, envelope
