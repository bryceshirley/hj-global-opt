
import numpy as np
import torch

from scipy.special import roots_hermite

from tabulate import tabulate

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from itertools import product

seed   = 30
torch.manual_seed(seed)


class HJ_MD_CD:
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
    def __init__(self, f, delta=100, t = 1e1, int_samples=1000, max_iters=1e4, f_tol = 1e-5,
                 distribution="Gaussian",beta=0.0, momentum=0.0,verbose=True,
                 line_search=True, stepsize=0.1, 
                 adaptive_delta=True, adaptive_delta_params=[1e-2,0.9,1.1],#0.9 before
                 adaptive_time=False, adaptive_time_params=[5e6,1e10,1.1,0.99,1.01]):
        
        
        # Algorithm Parameters
        self.delta            = delta
        self.t                = t
        self.int_samples      = int_samples
        self.distribution     = distribution
        self.f =f

        # Stopping Criteria
        self.max_iters        = max_iters
        self.f_tol            = f_tol

        ### Optional Parameters ###
        
        # Adaptive Delta
        self.adaptive_delta = adaptive_delta
        self.saturate_tol   = adaptive_delta_params[0]
        self.delta_minus     =  adaptive_delta_params[1]
        self.delta_plus      =  adaptive_delta_params[2]

        # Adaptive Time
        self.adaptive_time = adaptive_time
        self.t_min = adaptive_time_params[0]
        self.t_max = adaptive_time_params[1]
        self.theta = adaptive_time_params[2]
        self.t_minus = adaptive_time_params[3]
        self.t_plus = adaptive_time_params[4]

        # Line Search
        self.line_search = line_search
 

        z_line, weights_line = roots_hermite(int_samples)
        self.z_line = torch.tensor(z_line, dtype=torch.double)
        self.weights_line = torch.tensor(weights_line, dtype=torch.double)


        # Acceleration Parameters
        self.beta = beta
        self.momentum = momentum

        # Output Parameters
        self.verbose          = verbose
    


    def coordinate_prox(self,index,xk,deltak,tk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''

        xk_expanded = xk.expand(self.int_samples, self.n_features)
        z = xk_expanded.clone()


        rescale_factor = 1
        while True:
          # Apply Rescaling to time
          t_rescaled = tk/rescale_factor

          sigma = np.sqrt(2*deltak*t_rescaled)

          # Convert into f form
          z[:,index] = xk_expanded[:,index] - self.z_line*sigma

          f_values = self.f(z) # Size (int_samples,1)

          # Apply Rescaling to Exponent
          rescaled_exponent = rescale_factor*(-f_values/ deltak)   # shape =  n_samples

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
        alphak = - sigma*np.dot(w, self.z_line)

        #print(f"    Number of non-zero softmax weights on samples: {w[w>0.0].shape[0]}")

        
        return alphak

    def update_time(self, tk, rel_grad_uk_norm):
      '''
        time step rule

        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)
      '''

      if rel_grad_uk_norm <= self.theta:
        # Decrease t when relative gradient norm is smaller than theta
        tk = max(self.t_minus*tk, self.t_min)
      else:
        # Increas otherwise t when relative gradient norm is smaller than theta
        tk = min(self.t_plus*tk, self.t_max)

      return tk
    
    def run(self, x0):
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
        xk = x0.clone()
        tk = self.t
        deltak = self.delta
        yk = x0.clone()

        # Initialize History
        self.n_features = x0.shape[1]
        num_dimensions = self.n_features
        fk_hist = torch.zeros(self.max_iters+1)
        xk_hist = torch.zeros(self.n_features,self.max_iters+1)
        deltak_hist = torch.zeros(self.max_iters+1)
        tk_hist = torch.zeros(self.max_iters+1)
        fk = self.f(xk.view(1, self.n_features))
        xk_hist[:,0] = xk
        fk_hist[0] = fk
        deltak_hist[0] = deltak
        tk_hist[0] = tk

        # Define Outputs
        fmt = 'Epoch [{:3d}], Index [{:.3d}]: fk = {:6.2e} | alphak = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD-LS Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)

        epochs = self.max_iters

        f_epoch = self.f(xk.view(1, self.n_features))*1.1

        print(f'Epoch [{0:3d}]: fk = {fk.item():6.2e} | deltak = {deltak:6.2e}')

        alpha_k_minus = torch.zeros(num_dimensions)

        for k in range(epochs):
            shuffle_indices = torch.randperm(num_dimensions)

            f_epoch_new =  self.f(xk.view(1, self.n_features))

            if f_epoch < f_epoch_new:
                deltak*=0.8

            f_epoch = f_epoch_new
            
            non_zero_alphak = 0

            # momentum = 0.5
            # if k>0:
            #     yk = xk + momentum * (xk - xk_minus_1)
            # xk_minus_1 = xk.clone()
            # xk = yk.clone()
            if k > 0:
                valid_indices = [index for index in shuffle_indices if alpha_k_minus[index] != 0.0]
            else:  
                valid_indices = shuffle_indices

            print(len(valid_indices))

            
            for index in valid_indices:
                # if k > 0:
                #     if alpha_k_minus[index] == 0.0:
                #         alphak = self.coordinate_prox(index,xk,deltak,tk*3)
                #     else:
                #         alphak = self.coordinate_prox(index,xk,deltak,tk)
                # else:
                alphak = self.coordinate_prox(index,xk,deltak,tk)

                # if k == 0:

                if alphak != 0:
                    non_zero_alphak += 1
                else:
                    print(" ")
                    print(f"Before {alphak=}")
                    alphak = self.coordinate_prox(index,xk,deltak,tk*100)
                    print(f"After {alphak=}")
                    print(" ")
                
                alpha_k_minus[index] = alphak
                
                # beta = 0.0
                # first_moment[index] = beta*first_moment[index]+(1-beta)*alphak
   
                # Apply Gradient Descent
                xk[:,index] = xk[:,index] + alphak
                fk = self.f(xk.view(1, self.n_features))
            

                # Print Iteration Information
                # if self.verbose:
                #     print(f'Epoch [{k+1:3d}], Index [{index}]: fk = {fk.item():6.2e} | alphak = {alphak:6.2e}, | deltak = {deltak:6.2e}')
                #        # fmt.format(k+1,index.item(), fk.item(), alphak))

                # Update History
                xk_hist[:,k+1] = xk
                fk_hist[k+1] = fk
                deltak_hist[k+1] = deltak
                tk_hist[k+1] = tk

            # Stopping Criteria
            if fk < self.f_tol: # f value is less than tolerance
                if self.verbose:
                    print(f'    HJ-MAD converged to tolerence {self.f_tol:6.2e}')
                    print(f'    iter = {k}: f_value =  {fk}')
                break

            print(f'Epoch [{k+1:3d}]: fk = {fk.item():6.2e} | non_zero_alphak = {(non_zero_alphak/num_dimensions)*100:.3f}%, | deltak = {deltak:6.2e}')

    
        return xk, fk,xk_hist[:,0:k+1], fk_hist[0:k+1], deltak_hist[0:k+1],tk_hist[0:k+1], k+1

