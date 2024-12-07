
import numpy as np
import torch

from scipy.special import roots_hermite

from tabulate import tabulate

import matplotlib.pyplot as plt


seed   = 30
torch.manual_seed(seed)


# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------
# def weighted_softmax(F, w):
#     """
#     Compute the weighted softmax:
#         weights_i = ( exp(F_i - max(F_i))) / sum(exp(F_i - max(F_i)))
    
#     Parameters:
#         z (torch.Tensor): Input tensor of shape (n,).
#         a (torch.Tensor): Weights tensor of shape (n,). Must have the same shape as z.

#     Returns:
#         torch.Tensor: Weighted softmax output of shape (n,).
#     """
    # exp_z = torch.exp(F - torch.max(F))  # Shift z for numerical stability
    # weighted_exp = w * exp_z
    # norm = torch.sum(weighted_exp)
    # return weighted_exp / norm

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
    def __init__(self, f, f_numpy, f_name, x_true, delta=0.1, int_samples=1000,int_samples_line=150, t = 1e1, max_iters=5e4,
                 tol=5e-2, verbose=True,rescale0=1e-1,max_rescale= 1,saturate_tol=1e-9, line_integration_method='MC'):
      
        self.delta            = delta
        self.f                = f
        self.f_numpy          = f_numpy
        self.f_name           = f_name
        self.int_samples      = int_samples
        self.int_samples_line = int_samples_line
        self.max_iters        = max_iters
        self.tol              = tol
        self.t                = t
        self.x_true           = x_true
        self.verbose          = verbose
        self.rescale0         = rescale0
        self._rescale0         = rescale0
        self.saturate_tol     = saturate_tol
        self.line_integration_method = line_integration_method
        self.max_rescale      = max_rescale

        self.t_max = t*1e5

        self.n_features = x_true.shape[0] 

        if self.int_samples_line > 0 and self.line_integration_method == 'MC':
          self.weights = torch.ones(int_samples_line, dtype=torch.double)
        if self.int_samples_line > 0 and self.line_integration_method == 'GH':
          z_line, weights = roots_hermite(int_samples_line)
          self.z_line = torch.tensor(z_line, dtype=torch.double)
          self.weights = torch.tensor(weights, dtype=torch.double)

        #if self.verbose:
        self.print_parameters()
    
    def print_parameters(self):
        # Collect parameters into a list of tuples for tabulation
        def format_scientific(value):
            return f"{value:.3e}" if isinstance(value, (int, float)) else value
        
        parameters = [
            ("delta", self.delta),
            ("f_name", self.f_name),
            ("Dimensions/Features", self.n_features),
            ("int_samples", format_scientific(self.int_samples)),
            ("int_samples_line", format_scientific(self.int_samples_line)),
            ("max_iters", format_scientific(self.max_iters)),
            ("tol", self.tol),
            ("t", format_scientific(self.t)),
            ("rescale0", format_scientific(self.rescale0)),
            ("line_search", self.max_rescale),
            ("saturate_tol", format_scientific(self.saturate_tol)),
            ("integration_method", self.line_integration_method),
        ]
        
        # Print the table
        print(tabulate(parameters, headers=["Parameter", "Value"], tablefmt="pretty"))
    
    def h(self, tau, constants):
       
        direction, xk = constants

        xk_expanded = xk.expand(self.int_samples_line, self.n_features)
        direction_expanded = direction.expand(self.int_samples_line, self.n_features)

        y = xk_expanded + tau * direction_expanded
       
        return self.f(y)
    
    def improve_prox(self,xk, prox_xk, rescale_factor):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # print(f"{rescale_factor=:6.2e}")
        rescale_factor = 1 # self.rescale0
        delta         = 0.00001
        t = 100


        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0#-torch.norm(xk - prox_xk)

        constants = direction, xk

        iterations=0

        while True:
          # Apply Rescaling to time
          t_rescaled = t/rescale_factor

          sigma = np.sqrt(2*delta*t_rescaled)

          # Compute Function Values
          tau = tau_xk - self.z_line*sigma # Size (int_samples,1)
          h_values = self.h(tau.view(-1, 1),constants)

          # Apply Rescaling to Exponent
          rescaled_exponent = - rescale_factor*h_values/ self.delta

          # Remove Max Exponent to prevent Overflow
          max_exponent = torch.max(rescaled_exponent)  # Find the maximum exponent
          shifted_exponent = rescaled_exponent - max_exponent

          # Compute Exponential Term
          exp_term = torch.exp(shifted_exponent)

          # Compute the Denominator Integral
          w = self.weights*exp_term / np.dot(self.weights, exp_term)

          softmax_overflow = 1.0 - (w < np.inf).prod()
          if softmax_overflow and rescale_factor > 1e-15:
              rescale_factor /= 2
              iterations+=1
          else:
              break

          # v_delta = torch.sum(self.weights * exp_term)

          # # Compute Numerator Integral            
          # numerator = torch.sum(self.weights * self.z_line * exp_term)

        # Compute Gradient in 1D
        # grad_uk_L = (sigma/t_rescaled)*(numerator / v_delta)
        print(f"{iterations=}")
        grad_uk_L = (sigma/t_rescaled)*np.dot(w, self.z_line)


        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L

        prox_xk_new = xk + prox_tau*direction

        #self.plot_1d_prox(xk.view(1, self.n_features).numpy(), prox_xk.numpy(), prox_xk_new.numpy(), t_rescaled)

        
        return prox_xk_new


    def compute_directional_prox(self,xk):
        """
        Compute the prox of f using Monte Carlo Integration.
        
        Parameters:
            x (float): Point at which to compute the prox.
            t (float): Time parameter.
            delta (float): Viscousity parameter.
        Returns:
            prox (float): Computed prox.
        """
        t     = self.t
        delta = self.delta

        rescale_factor = self.rescale0
        scale_minus = 0.9
        scale_plus = 1.1
        min_rescale=1e-10
        max_rescale= self.max_rescale
        iterations = 0

        # Reshape xk to ensure broadcasting
        # xk_expanded = xk.expand(self.int_samples_line, self.n_features)
        # z_expanded = self.z_line.expand(self.int_samples_line, self.n_features)
        # Reshape xk to ensure broadcasting
# Ensure xk has the correct shape for broadcasting (int_samples, n_features)
        xk_squeezed = xk.squeeze(0)  # Remove unnecessary dimensions (if xk has shape [1, 1, n_features])
        xk_expanded = xk_squeezed.expand(self.int_samples_line, self.n_features)  # Shape: (int_samples_line, n_features)

        # Reshape z_line to allow broadcasting, then expand
        z_expanded = self.z_line.unsqueeze(1)  # Shape: (int_samples, 1)
        z_expanded = z_expanded.expand(self.int_samples_line, self.n_features)  # Shape: (int_samples_line, n_features)
        while True:
            # Apply Rescaling to time
            t_rescaled = t/rescale_factor

            sigma = np.sqrt(2*delta*t_rescaled)

            # Compute Perturbed Points
            y = xk_expanded - (sigma * z_expanded)

            # Compute Function Values
            f_values = self.f(y)  # Assume f supports batch processing, returns shape (int_samples,)
            rescaled_exponent = -rescale_factor * f_values / delta
            exp_term = torch.exp(rescaled_exponent)

            # print(f"{self.weights.shape=}")
            # print(f"{exp_term.shape=}")

            # Compute weights
            denominator = torch.dot(self.weights, exp_term)  # Scalar
            w = torch.where(
                denominator != 0,
                self.weights * exp_term / denominator,
                torch.full_like(self.weights, float("inf")),
            )

            # Check for Overflow
            softmax_overflow = 1.0 - (w < np.inf).prod()
            #softmax_overflow = 1.0 - np.prod((w.cpu().numpy() < np.inf).astype(float))

            if softmax_overflow:
                # Adjust rescale factor and increment iteration count
                rescale_factor = max(scale_minus*rescale_factor, min_rescale)
                iterations += 1
            else:
                break
        
        # Increase intial rescale factor if it is too small to better utilize samples
        if iterations == 0:
            rescale_factor = min(scale_plus*rescale_factor, max_rescale)
        
        self.rescale0 = rescale_factor

        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()

        print(f"{prox_xk=}")

        return prox_xk, rescale_factor

        # # Sample From a Guassian with mean x and standard deviation
        # standard_dev = np.sqrt(delta*t/rescale_factor)

        # z = torch.randn(self.int_samples,self.n_features)

        # y = standard_dev * z + xk

        # f_values = self.f(y)

        # rescaled_exponent = -rescale_factor*f_values/delta
        
        # w = torch.softmax(rescaled_exponent, dim=0) # shape = n_samples 
    
        # softmax_overflow = 1.0 - (w < np.inf).prod()
        # if softmax_overflow:
        #   # Adjust rescale factor and increment iteration count
        #   rescale_factor = max(scale_minus*rescale_factor, min_rescale)
        #   iterations += 1
        # else:
        #   break



    
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
        xk = x0.clone()
        x_opt = xk.clone()
        fk_old = self.f(x_opt.view(1,-1))

        # Initialize History
        xk_hist = torch.zeros(self.max_iters+1, self.n_features)
        xk_error_hist = torch.zeros(self.max_iters+1)
        fk_hist = torch.zeros(self.max_iters+1)


        # Find Initial "Best Direction"
        prox_xk, rescale_factor  = self.compute_directional_prox(xk)
        f_prox = self.f(prox_xk.view(1, self.n_features))

        # Define Outputs
        fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} '
        if self.verbose:
            print('-------------------------- RUNNING Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)

        successful_ls_count = 0
        for k in range(self.max_iters):
            # Update History
            xk_hist[k, :] = xk
            xk_error_hist[k] = torch.norm(xk - self.x_true)
            fk_hist[k] = f_prox

            if self.verbose:
                print(fmt.format(k+1, fk_hist[k], xk_error_hist[k]))

            # Find Proximal in 1D Along this Direction
            if self.int_samples_line > 0:
              prox_xk_new = self.improve_prox(xk,prox_xk, rescale_factor)
              f_prox_new = self.f(prox_xk_new.view(1, self.n_features))
              if f_prox_new < f_prox:
                print("Improvement from line search")
                successful_ls_count += 1
                prox_xk = prox_xk_new
              else:
                print("No improvement in line search")


            xk = prox_xk
            fk = self.f(xk.view(1, self.n_features))
            
            # Find Initial "Directional Prox in R^n"
            prox_xk, rescale_factor  = self.compute_directional_prox(xk)
            f_prox = self.f(prox_xk.view(1, self.n_features))

            # Only Update x_opt if the new point is better in f
            if fk < fk_old:
                x_opt = xk.clone()
            fk_old = fk

            # Stopping Criteria
            opt_error = torch.norm(x_opt - self.x_true)
            if xk_error_hist[k] < self.tol:
                if self.verbose:
                    print(f'HJ-MAD converged to tolerence {self.tol:6.2e}')
                    print(f'iter = {k}, Error =  {opt_error:6.2e}')
                break
            if k > 0 and np.abs(torch.norm(xk_hist[k] - xk_hist[k-1])) < self.saturate_tol*torch.norm(xk_hist[k-1]): 
                if self.verbose:
                    print('HJ-MAD converged due to saturation')
                    print(f'iter = {k}, Error =  {opt_error:6.2e}')
                # self.t = min(2*self.t , self.t_max)
                # rescale_factor = self._rescale0
                # test_t = min(2*self.t , self.t_max)
                # prox_xk, rescale_factor = self.compute_directional_prox(xk,test_t)
                # prox_xk_test = self.improve_prox(xk,prox_xk, rescale_factor)
                # f_prox_test = self.f(prox_xk_test.view(1, self.n_features))
                # if f_prox_test < f_prox:
                #    t = min(2*self.t , self.t_max)
                # else:
                #    t = max(0.6*self.t , self.t_min)
            # if k > 50 and torch.std(fk_hist[k-50:k+1]) < self.saturate_tol*fk_hist[k]:
            #     if self.verbose:
            #         print('HJ-MAD begun to oscillate')
            #         print(f'iter = {k}, Error =  {opt_error:6.2e}')
            #     break
            
            print(f"Rescale Factor = {self.rescale0:6.2e}")
            print(f"Time = {self.t:6.2e}, Max Time = {self.t_max:6.2e}")


        successful_ls_portion = successful_ls_count/(k+1)
        return x_opt, xk_hist[:k+1,:], xk_error_hist[:k+1], fk_hist[:k+1], successful_ls_portion
    
    def plot_1d_prox(self, xk, prox_xk_old, prox_xk_new, tk, num_points=1000):
        """
        Plots the 1D descent for the current dimension.

        Args:
            xk (torch.Tensor): Current position.
            dim (int): The current dimension being optimized.
            domain (tuple): The range over which to vary the current dimension.
            num_points (int): Number of points to sample in the domain.
        """

        tauk = 0
        direction = (xk - prox_xk_old)/np.linalg.norm(xk - prox_xk_old)

        tau_old = np.dot(prox_xk_old - xk, direction.T)
        tau_new = np.dot(prox_xk_new - xk, direction.T)
        f_vals = []
        h_vals = []

        tau_vals = np.linspace(-40, 20, num_points)
        # Create x_vals with shape (num_points, 2)
        x_vals = np.zeros((num_points, self.n_features))

        # Compute x_vals by adding tau * direction for each component of x0
        for i in range(self.n_features):
            x_vals[:, i] = xk[0,i] + tau_vals.T * direction[0,i]


        for x in x_vals:
            xk_varied = x
            f_vals.append(self.f_numpy(xk_varied))
            h_vals.append(self.f_numpy(xk_varied) + 1/(2*tk)*np.linalg.norm(xk_varied-xk)**2)


        # print(f"{self.delta} * {self.t_vec[0]} = {self.delta * self.t_vec[0]}")
        # print(f'Std Dev: {std_dev}, Std Dev Minus: {std_dev_minus}, Std Dev Plus: {std_dev_plus}')

        plt.figure()
        plt.plot(tau_vals, f_vals, '-', color='black', label=f'f(x)')
        plt.plot(tau_vals, h_vals, '-', color='blue', label=r'$f(x) + \frac{1}{2t_0} ||x - x_0||^2$')
        #plt.plot(tauk, self.f_numpy(xk[0,:]), '*', color='black', label=f'xk')
        plt.plot(tau_old, self.f_numpy(prox_xk_old[0,:]), '*', color='green', label=f'Full Space')
        plt.plot(tau_new, self.f_numpy(prox_xk_new[0,:]), '*', color='red', label=f'On Line')

        plt.xlabel(f'Tau')
        plt.ylabel('Function Value')
        plt.title(f'1D Descent for Chosen Line')
        plt.legend()
        plt.show()
