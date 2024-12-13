
import numpy as np
import torch

from scipy.special import roots_hermite

from tabulate import tabulate

import matplotlib.pyplot as plt
from itertools import product

seed   = 30
torch.manual_seed(seed)


class HJ_MD_LS:
    ''' 
        Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
        problems via a zeroth-order sampling scheme.
        
        Inputs:
          1)  f            = function to be minimized. Inputs have size (n_samples x n_features). Outputs have size n_samples
          2)  x_true       = true global minimizer
          3)  delta        = coefficient of viscous term in the HJ equation
          4)  int_samples  = number of samples used to approximate expectation in heat equation solution
          5)  x_true       = true global minimizer
          6)  t_vec        = time vector containig [initial time, minimum time allowed, maximum time]
          7)  max_iters    = max number of iterations
          8)  tol          = stopping tolerance
          9)  theta        = parameter used to update tk
          10) beta         = exponential averaging term for gradient beta (beta multiplies history, 1-beta multiplies current grad)
          11) eta_vec      = vector containing [eta_minus, eta_plus], where eta_minus < 1 and eta_plus > 1 (part of time update)
          11) alpha        = step size. has to be in between (1-sqrt(eta_minus), 1+sqrt(eta_plus))
          12) fixed_time   = boolean for using adaptive time
          13) verbose      = boolean for printing
          14) momentum     = For acceleration.
          15) accelerated  = boolean for using Accelerated Gradient Descent

        Outputs:
          1) x_opt                    = optimal x_value approximation
          2) xk_hist                  = update history
          3) tk_hist                  = time history
          4) fk_hist                  = function value history
          5) xk_error_hist            = error to true solution history 
          6) rel_grad_uk_norm_hist    = relative grad norm history of Moreau envelope
    '''
    def __init__(self, delta=100, t = 1e1, int_samples=1000, max_iters=1e4, f_tol = 1e-5,
                 sat_tol=1e-5,distribution="Gaussian",delta_dampener=0.99,beta=0.5,verbose=True):
        # Algorithm Parameters
        self.delta            = delta
        self.t                = t
        self.int_samples      = int_samples

        self.delta_dampener =delta_dampener
        self.beta = beta
        self.distribution = distribution

        # Stopping Criteria
        self.saturate_tol     = sat_tol
        self.max_iters        = max_iters
        self.f_tol            = f_tol

        self.verbose          = verbose

        # Generate Hermite Quadrature Points in R^1
        z_line, weights_line = roots_hermite(int_samples)
        self.z_line = torch.tensor(z_line, dtype=torch.double)
        self.weights_line = torch.tensor(weights_line, dtype=torch.double)

    def f_line(self, tau, xk, direction):

        xk_expanded = xk.expand(self.int_samples, self.n_features)
        direction_expanded = direction.expand(self.int_samples, self.n_features)

        y = xk_expanded + tau * direction_expanded
       
        return self.f(y)
    
    def improve_prox_with_line_search(self,xk, prox_xk,deltak):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0#-torch.norm(xk - prox_xk)

        rescale_factor=1
        while True:
          # Apply Rescaling to time
          t_rescaled = self.t/rescale_factor

          sigma = np.sqrt(2*deltak*t_rescaled)

          # Compute Function Values
          tau = tau_xk - self.z_line*sigma # Size (int_samples,1)
          h_values = self.f_line(tau.view(-1, 1),xk, direction) # Size (int_samples,1)

          # Apply Rescaling to Exponent
          rescaled_exponent = - rescale_factor*h_values/ deltak

          # Remove Max Exponent to prevent Overflow
          shifted_exponent = rescaled_exponent - torch.max(rescaled_exponent)

          # Compute Exponential Term
          exp_term = torch.exp(shifted_exponent).squeeze().double()

          # Compute the Denominator Integral
          # Compute weights_line
          denominator = torch.dot(self.weights_line, exp_term)  # Scalar
          w = torch.where(
            denominator != 0,
            self.weights_line * exp_term / denominator,
            np.inf,#torch.full_like(self.weights, float("inf")),
          )

          # Check for Overflow
          softmax_overflow = 1.0 - (w < np.inf).prod()
          if softmax_overflow and rescale_factor > 1e-20:
              rescale_factor /= 2
          else:
              break
        
        # Compute Prox on Line
        grad_uk_L = (sigma/t_rescaled)*np.dot(w, self.z_line)

        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L
        prox_xk_new = xk + prox_tau*direction
        
        return prox_xk_new
    
    def compute_prox(self,xk,deltak):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed
        rescale_factor = 1
        iterations = 0

        xk = xk.squeeze(0)  # Remove unnecessary dimensions
        xk_expanded = xk.expand(self.int_samples, self.n_features) 

        while True:
            # Apply Rescaling to time
            t_rescaled = self.t/rescale_factor

            standard_deviation = np.sqrt(deltak*t_rescaled)

            # Compute Perturbed Points
            if self.distribution == "Cauchy":
                cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=standard_deviation)

                # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
                y = cauchy_dist.sample((self.int_samples,))
            else:
                y = xk_expanded + standard_deviation*torch.randn(self.int_samples, self.n_features)

            # Compute Function Values
            f_values = self.f(y)  
            rescaled_exponent = -rescale_factor * f_values / deltak
            rescaled_exponent = rescaled_exponent - torch.max(rescaled_exponent)
            exp_term = torch.exp(rescaled_exponent)

            # Compute weights_line
            denominator = torch.sum(exp_term)  # Scalar
            w = torch.where(
                denominator != 0,
                exp_term / denominator,
                np.inf,#torch.full_like(self.weights, float("inf")),
            )

            # Check for Overflow
            softmax_overflow = 1.0 - (w < np.inf).prod()

            if softmax_overflow and rescale_factor > 1e-200:
                # Adjust rescale factor and increment iteration count
                rescale_factor /= 2
                iterations += 1
            else:
                break
        
        self.rescale0 = rescale_factor

        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()
        f_prox = self.f(prox_xk).item()
        f_xk = self.f(xk.expand(1, self.n_features)).item()

        # Update delta if the proximal point is worse than the current point
        if self.verbose:
            print(f"    Samples taken into account by softmax: {w[w>0.0].shape[0]}")
            print(f"    f(prox): {f_prox} | f(xk): {f_xk}")
        if f_prox > f_xk:
            deltak = deltak*self.delta_dampener

        # Improve the proximal point using line search
        prox_xk_new = self.improve_prox_with_line_search(xk,prox_xk,deltak)
        f_prox_new = self.f(prox_xk_new.view(1, self.n_features))
        if f_prox_new < f_prox:
            prox_xk = prox_xk_new
            if self.verbose:
                print(f"    Improvement from line search | f(prox_ls): {f_prox_new}")
        else:
            if self.verbose:
                print(f"    No improvement from line search | f(prox_ls): {f_prox_new}")

        # Return the proximal point for xk
        return prox_xk, deltak

    def run(self, f, x0):
        """
        Runs the coordinate descent optimization process.

        Args:
            x0 (torch.Tensor): Initial guess for the minimizer.
            num_cycles (int): Number of cycles to run the coordinate descent.

        Returns:
            torch.Tensor: Optimal solution found by the coordinate descent.
            list: History of solutions for each cycle.
            list: History of the entire optimization process.
            list: Error history for each cycle.
        """
        # Initialize Variables
        # Function Parameter
        self.f = f
        xk = x0.clone()
        x_opt = x0.clone()
        self.n_features = x0.shape[1]
        deltak = self.delta

        # Initialize History
        fk_hist = torch.zeros(self.max_iters+1)
        #delta_hist = torch.zeros(self.max_iters+1)
        f_k = self.f(xk.view(1, self.n_features))
        fk_hist[0] = f_k
        #delta_hist[0] = self.delta

        # Define Outputs
        fmt = '[{:3d}]: fk = {:6.2e} | delta = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD-LS Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)
            print(fmt.format(0, fk_hist[0], deltak))

        for k in range(self.max_iters):
            # Compute Proximal Point and Function Value
            prox_xk,deltak = self.compute_prox(xk,deltak)
            f_prox = self.f(prox_xk.view(1, self.n_features))

            # Update Optimal xk
            if f_prox < f_k:
                x_opt = prox_xk


            # Stopping Criteria
            if f_prox < self.f_tol: # f value is less than tolerance
                if self.verbose:
                    print(f'    HJ-MAD converged to tolerence {self.f_tol:6.2e}')
                    print(f'    iter = {k}: f_value =  {f_prox}')
                break
            # if k > 0: # Check for saturation
            #     relative_gradient_error = np.abs(np.abs(torch.norm(xk)/torch.norm(prox_xk))-1)
            #     if relative_gradient_error < self.saturate_tol:
            #         if self.verbose:
            #             print('HJ-MAD converged due to error saturation')
            #             print(f"Relative Gradient Error: {relative_gradient_error} | saturate_tol: {self.saturate_tol}")
            #         break

            # Update xk
            if k == 0:
                first_moment = prox_xk

            # Update xk
            first_moment = self.beta*first_moment+(1-self.beta)*(xk-prox_xk)
            xk = xk - first_moment
            f_k = self.f(xk.view(1, self.n_features))

            # Update History
            fk_hist[k+1] = f_k
            #delta_hist[k+1] = self.delta

            if self.verbose:
                print(fmt.format(k+1, fk_hist[k+1], deltak))

        return x_opt, k+1 #delta_hist[:k+1]
