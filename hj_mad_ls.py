
import numpy as np
import torch

from scipy.special import roots_hermite
from torch.distributions.studentT import StudentT

from tabulate import tabulate

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
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
                 distribution="Gaussian",beta=0.0, momentum=0.0,verbose=True,
                 line_search=True, stepsize=1.0, cooling=0.99,
                 adaptive_delta=True, adaptive_delta_params=[1e-2,0.9,1.1],#0.9 before
                 adaptive_time=False, adaptive_time_params=[5e-5,1e5,1.1,0.99,1.01], dof=2.0):
        
        
        # Algorithm Parameters
        self.delta            = delta
        self.t                = t
        self.int_samples      = int_samples
        self.distribution     = distribution

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
        if line_search:
            self.stepsize = 1.0 # Line search finds the optimal step size
            # Generate Hermite Quadrature Points in R^1
            z_line, weights_line = roots_hermite(int_samples)
            self.z_line = torch.tensor(z_line, dtype=torch.double)
            self.weights_line = torch.tensor(weights_line, dtype=torch.double)
        else:
            self.stepsize = stepsize

        # Acceleration Parameters
        self.beta = beta
        self.momentum = momentum

        self.dof = dof
        self.cooling = cooling

        # Output Parameters
        self.verbose          = verbose
    


    def improve_prox_with_line_search(self,xk, prox_xk,deltak,tk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''

        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0#-torch.norm(xk - prox_xk)

        xk_expanded = xk.expand(self.int_samples, self.n_features)
        direction_expanded = direction.expand(self.int_samples, self.n_features)

        rescale_factor=1
        # while True:
          # Apply Rescaling to time
        t_rescaled = tk/rescale_factor

        sigma = np.sqrt(2*deltak*t_rescaled)

        # Compute Function Values
        tau = tau_xk - self.z_line*sigma # Size (int_samples,1)

        # Convert into f form
        y = xk_expanded + tau.view(-1, 1) * direction_expanded
        f_values = self.f(y) # Size (int_samples,1)

        # Apply Rescaling to Exponent
        rescaled_exponent = - rescale_factor*f_values/ deltak 

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

        #   # Check for Overflow
        #   softmax_overflow = 1.0 - (w < np.inf).prod()
        #   if softmax_overflow and rescale_factor > 1e-20:
        #       rescale_factor /= 2
        #   else:
        #       break
        
        # Compute Prox on Line
        grad_uk_L = (sigma/t_rescaled)*np.dot(w, self.z_line)

        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L
        prox_xk_new = xk + prox_tau*direction
        
        return prox_xk_new
    
    def compute_prox(self,xk,deltak,tk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed
        rescale_factor = 1
        iterations = 0

        xk = xk.squeeze(0)  # Remove unnecessary dimensions
        xk_expanded = xk.expand(self.int_samples, self.n_features) 

        # while True:
        #     # Apply Rescaling to time
        #     t_rescaled = tk/rescale_factor

        # Compute Perturbed Points
        if self.distribution == "Cauchy" and tk > 0:
            scale = np.sqrt(2*deltak*tk)
            cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=scale)

            y = cauchy_dist.sample((self.int_samples,))
            f_values = self.f(y)  

            two_norm_squared = torch.sum((y - xk_expanded)**2, dim=1)
            rescaled_exponent = -(1/deltak)*(f_values + (1/2*tk)*two_norm_squared - deltak*torch.log(two_norm_squared/(scale**2)+1))
        else:
            # if deltak*tk < 5e-3:
            #     standard_deviation = np.sqrt(1e-6)
            # else:
            standard_deviation = np.sqrt(deltak*tk)
            y = xk_expanded + standard_deviation*torch.randn(self.int_samples, self.n_features)

            f_values = self.f(y)  
            rescaled_exponent = -rescale_factor * f_values / deltak

        # Compute Function Values
        rescaled_exponent = rescaled_exponent - torch.max(rescaled_exponent)
        exp_term = torch.exp(rescaled_exponent)

        # Compute weights_line
        denominator = torch.sum(exp_term)  # Scalar
        w = torch.where(
            denominator != 0,
            exp_term / denominator,
            np.inf,#torch.full_like(self.weights, float("inf")),
        )

            # # Check for Overflow
            # softmax_overflow = 1.0 - (w < np.inf).prod()

            # if softmax_overflow and rescale_factor > 1e-200:
            #     # Adjust rescale factor and increment iteration count
            #     rescale_factor /= 2
            #     iterations += 1
            # else:
            #     break

        prox_xk = torch.matmul(w.t(), y)

        # Update delta if the proximal point is worse than the current point
        #if self.verbose:
        #    print(f"    Number of non-zero softmax weights on samples: {w[w>0.0].shape[0]}")

        # Return the proximal point for xk
        return prox_xk

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
        xk_minus_1 = x0.clone()
        tk = self.t
        deltak = self.delta

        # Initialize History
        self.n_features = x0.shape[1]
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
        fmt = '[{:3d}]: fk = {:6.2e} | deltak = {:6.2e} | tk = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD-LS Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)
            print(fmt.format(0, fk_hist[0], deltak, tk))

        saturation_count = 0

        for k in range(self.max_iters):
            # Compute Proximal Point
            prox_xk = self.compute_prox(xk,deltak,tk)
            f_prox = self.f(prox_xk.view(1, self.n_features))

            # Update Delta
            if self.adaptive_delta:
                # Dampen delta if the proximal point is worse than the current point
                if f_prox >= 1.01*fk: #1.1*fk:
                    if self.verbose:
                        print(f"    f(xk): {fk.item():.5e} | f(prox): {f_prox.item():.5e}")
                    deltak *= self.delta_minus
    
                # Prevent delta from becoming too small
                elif saturation_count > 5: # Check for saturation
                    saturation_count = 0
                    relative_gradient_error = torch.abs(torch.abs(torch.norm(fk_hist[k] )/torch.norm(fk_hist[k-5] ))-1)
                    if relative_gradient_error < self.saturate_tol:
                        deltak *= self.delta_plus
                        if self.verbose:
                            print(f"    Delta increased to {deltak}")
                saturation_count += 1

            # Line Search
            if self.line_search:
                prox_xk_new = self.improve_prox_with_line_search(xk,prox_xk,deltak,tk)
                f_prox_new = self.f(prox_xk_new.view(1, self.n_features)).item()
                if f_prox_new < 0.9*f_prox:
                    prox_xk = prox_xk_new
                    f_prox = f_prox_new
                    if self.verbose:
                        print(f"    Improvement from line search | f(prox_ls): {f_prox_new:.5e}")
                else:
                    if self.verbose:
                        print(f"    No improvement from line search | f(prox_ls): {f_prox_new:.5e}")

            # Momentum (yk = xk if momentum = 0)
            yk = xk + self.momentum * (xk - xk_minus_1)
            xk_minus_1 = xk

            # Update first moment (first_moment = (yk-prox_xk) if beta = 0)
            if k == 0:
                first_moment = prox_xk
                

            first_moment = self.beta*first_moment+(1-self.beta)*(yk-prox_xk)

            # Apply Gradient Descent
            xk = yk - self.stepsize*first_moment
            fk = self.f(xk.view(1, self.n_features))
            
            # Update time
            if self.adaptive_time:
                grad_norm = torch.norm(first_moment)
                if k >0:
                    rel_grad_norm = grad_norm/(grad_norm_old + 1e-12)
                    tk = self.update_time(tk,rel_grad_norm)
                grad_norm_old = grad_norm

            # Print Iteration Information
            if self.verbose:
                print(fmt.format(k+1, fk.item(), deltak, tk))

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

        
        if k == self.max_iters-1:
            if self.verbose:
                print(f"    HJ-MAD did not converge after {self.max_iters} iterations")
    
        return xk, fk,xk_hist[:,0:k+1], fk_hist[0:k+1], deltak_hist[0:k+1],tk_hist[0:k+1], k+1

    def HJ_simulated_annealing(self, f, x0):
        self.f = f
        xk = x0.clone()
        delta0 = self.delta
        xk_minus_1 = x0.clone()
        cooling_rate = self.cooling

        # Initialize History
        self.n_features = x0.shape[1]
        fk_hist = torch.zeros(self.max_iters+1)
        xk_hist = torch.zeros(self.n_features,self.max_iters+1)
        deltak_hist = torch.zeros(self.max_iters+1)
        p_hist = torch.zeros(self.max_iters+1)
        fk = self.f(xk.view(1, self.n_features))
        xk_hist[:,0] = xk
        fk_hist[0] = fk 
        deltak_hist[0] = delta0
        p_hist[0] = 0
        p=0

        # Define Outputs
        fmt = '[{:3d}]: Energy fk = {:6.2e} | Temperature deltak = {:6.2e} | Time tk = {:6.2e} | Brownian Variance = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD-LS Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)
            # print(fmt.format(0, fk_hist[0], delta0, 0, 0))

        saturation_count = 0
        restarts =0
        fk_opt = fk
        xk_opt = xk
        tk =1
        while True:
            # accepted_moves =0
            while True:
                if tk >= self.max_iters-1:
                    break
                # Apply Cooling Schedule
                #deltak = self.cooling*deltak # Geometric Cooling
                deltak = delta0/(1+self.cooling*(tk**3))
                #deltak = delta0/np.log(tk+np.e) # Classic Cooling

                # Approximate Brownian / Boltzmann Expectation
                if self.distribution == "Cauchy":
                    #scale = np.sqrt(2*deltak*tk)
                    prox_xk = self.compute_prox(xk,deltak,tk)# torch.distributions.Cauchy(loc=xk, scale=scale).sample()
                else: # Gaussian
                    #standard_deviation = np.sqrt(deltak*tk)
                    prox_xk = self.compute_prox(xk,deltak,tk)#prox_xk = xk + standard_deviation*torch.randn(1, self.n_features)

                if self.line_search and tk > 0:
                    prox_xk = self.improve_prox_with_line_search(xk,prox_xk,deltak,tk)

                delta_f = self.f(prox_xk.view(1, self.n_features)) - fk

                # # Apply Metropolis-Hastings Correction
                p = torch.exp(-delta_f / deltak)
                if delta_f < 0 or p > np.random.rand():
                    # accepted_moves += 1
                    xk = prox_xk
                    fk = self.f(xk.view(1, self.n_features))


                # # Calculate acceptance rate
                # acceptance_rate = accepted_moves / tk
                
                # # Adjust the temperature based on the acceptance rate
                # if acceptance_rate < 0.2:
                #     # Accelerate the cooling rate to focus on convergence
                #     cooling_rate *= 1.1

                # Apply Accelerated Gradient Descent
                yk = xk + self.momentum * (xk - xk_minus_1)
                xk_minus_1 = xk

                # Update first moments (Weight Sum Moving Average) (first_moment = (yk-prox_xk) if beta = 0)
                if tk == 1:
                    first_moment = prox_xk

                first_moment = self.beta*first_moment+(1-self.beta)*(yk-prox_xk)

                # Apply Moreau Envelope Descent
                xk = yk - self.stepsize*first_moment
                fk = self.f(xk.view(1, self.n_features))

                if fk <fk_opt:
                    fk_opt = fk
                    xk_opt = xk

                # Print Iteration Information
                if tk % 100 == 1:
                    if self.verbose:
                        print(fmt.format(tk, fk.item(), deltak, tk,deltak*tk))
                # Update History
                xk_hist[:,tk+1] = xk
                fk_hist[tk+1] = fk
                deltak_hist[tk+1] = deltak
                p_hist[tk+1] = p

                # Stopping Criteria
                if fk < self.f_tol: # f value is less than tolerance
                    if self.verbose:
                        print(f'    HJ-MAD converged to tolerence {self.f_tol:6.2e}')
                        print(f'    iter = {tk}: f_value =  {fk}')
                    break

                # Check for Relative Saturation
                relative_saturation = abs(delta_f) / abs(fk) # Avoid division by zero
                if relative_saturation < 1e-3:
                    saturation_count += 1
                else:
                    saturation_count = 0  # Reset if progress resumes

                # Trigger Restart if Saturation Detected
                if saturation_count >= 10:
                    if self.verbose:
                        print(f"Restart triggered at iteration {tk} due to relative saturation.")
                    delta0 = delta0*1.1
                    cooling_rate *= 0.9
                    #xk = xk_opt + torch.randn_like(xk) * 0.01  # Perturb xk slightly
                    fk = self.f(xk.view(1, self.n_features))  # Recompute objective value
                    print(fmt.format(tk, fk.item(), deltak, tk,deltak*tk))
                    saturation_count = 0  # Reset saturation count
                    restarts += 1
                    break  # Skip further updates in the current iteration
                tk += 1

            if tk >= self.max_iters-1:
                if self.verbose:
                    print(f"    HJ-MAD did not converge after {tk} iterations")
                break  
            if fk < self.f_tol: 
                break  
        
        return xk_opt, fk_opt,xk_hist[:,2:tk+1], fk_hist[2:tk+1], deltak_hist[2:tk+1],p_hist[2:tk+1], tk+1
