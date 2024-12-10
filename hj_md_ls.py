
import numpy as np
import torch

from scipy.special import roots_hermite

from tabulate import tabulate

import matplotlib.pyplot as plt
from itertools import product

seed   = 30
torch.manual_seed(seed)


# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------
# def weighted_softmax(F, w):
#     """
#     Compute the weighted softmax:
#         weights_line_i = ( exp(F_i - max(F_i))) / sum(exp(F_i - max(F_i)))
    
#     Parameters:
#         z (torch.Tensor): Input tensor of shape (n,).
#         a (torch.Tensor): weights_line tensor of shape (n,). Must have the same shape as z.

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
    def __init__(self, f, f_numpy, f_name, x_true, delta=0.1, t = 1e1, int_samples=1000,int_samples_line=150, max_iters=5e4,
                 tol=5e-2,rescale0=1e-1,saturate_tol=1e-9,integration_method="MC",delta_dampener=0.8,verbose=True,plot=False):
      
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
        self.saturate_tol     = saturate_tol
        self.plot             = plot
        self.integration_method = integration_method
        self.distribution = "Cauchy"
        self.delta_dampener =delta_dampener


        self.n_features = x_true.shape[0] 

        if self.verbose:
            self.print_parameters()

        if int_samples_line > 0:
            # Generate Hermite Quadrature Points in R^1
            z_line, weights_line = roots_hermite(int_samples_line)
            self.z_line = torch.tensor(z_line, dtype=torch.double)
            self.weights_line = torch.tensor(weights_line, dtype=torch.double)
        
        if integration_method == "GH":

            # Generate Hermite Quadrature Points in R^n
            self.weights, self.z, self.filtered_samples = self.generate_gh_grid_matrix_with_threshold()

    def generate_gh_grid_matrix_with_threshold(self): #sample_fraction=0.01):
        """
        Generates a column matrix for n-dimensional Gauss-Hermite quadrature
        and removes rows with weights below a threshold.

        Args:
            rho (float): Correlation coefficient.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: A tensor of size (filtered_samples,) with weight products.
                - torch.Tensor: A tensor of size (filtered_samples, n) with the corresponding points.
        """
        # n (int): Number of dimensions.
        # m (int): Number of Gauss-Hermite nodes per dimension.
        m = int(self.int_samples)
        n = int(self.n_features)
        # Generate 1D Gauss-Hermite nodes and weights
        nodes, weights = roots_hermite(m)

        # Determine the total number of samples
        #total_samples = int(sample_fraction * m**n)

        # Calculate threshold θm
        w_1 = weights[0]
        w_mid = weights[m // 2] if m % 2 == 1 else weights[m // 2 - 1]
        theta_m = w_1 * w_mid / m

        # Create correlation matrix (all off-diagonal elements set to rho)

        # Cartesian product for n-dimensional indices
        indices = product(range(m), repeat=n)

        # Initialize lists to store the results
        weight_products = []
        grid_points = []

        for idx in indices:
            # Extract nodes and weights for the current indices
            z = np.array([nodes[i] for i in idx])  # Corresponding z values
            w = [weights[i] for i in idx]  # Corresponding weights
            w_prod = np.prod(w)  # Product of weights

            # Apply the threshold
            if w_prod <= theta_m:
                continue

            # Append the weight product and the corresponding point
            weight_products.append(w_prod)
            grid_points.append(z)

        # Convert weight_products list to a numpy array
        weight_products_array = np.array(weight_products, dtype=np.float64)

        # Convert grid_points list of arrays to a single numpy array
        grid_points_array = np.array(grid_points, dtype=np.float64)

        # Convert to PyTorch tensors
        weight_products_tensor = torch.tensor(weight_products_array, dtype=torch.double)
        grid_points_tensor = torch.tensor(grid_points_array, dtype=torch.double)

        # Randomly remove 80% of the samples
        num_samples = weight_products_tensor.shape[0]
        num_samples_to_remove = int(0.0 * num_samples)

        # Get random indices to keep 20% of the samples
        keep_indices = np.random.choice(num_samples, num_samples - num_samples_to_remove, replace=False)

        # Filter the weight products and grid points using the selected indices
        weight_products_tensor = weight_products_tensor[keep_indices]
        grid_points_tensor = grid_points_tensor[keep_indices]

        filtered_samples = weight_products_tensor.shape[0]

        if self.verbose:
            print(f"\nGauss-Hermite Quadrature Grid Matrix in R^n with Threshold:")
            print(" - Percentage of Retained Points: {:.2f}%".format((filtered_samples / m**n) * 100))
            print(" - Number of samples with weights above threshold:", filtered_samples)
            print("")

        return weight_products_tensor, grid_points_tensor, filtered_samples
    
    def print_parameters(self):
        # Collect parameters into a list of tuples for tabulation
        def format_scientific(value):
            return f"{value:.3e}" if isinstance(value, (int, float)) else value
        
        parameters = [
            ("f_name", self.f_name),
            ("Dimensions/Features", self.n_features),
            ("delta", self.delta),
            ("t", format_scientific(self.t)),
            ("int_samples", format_scientific(self.int_samples)),
            ("int_samples_line", format_scientific(self.int_samples_line)),
            ("max_iters", format_scientific(self.max_iters)),
            ("tol", self.tol),
            ("rescale0", format_scientific(self.rescale0)),
            ("saturate_tol", format_scientific(self.saturate_tol)),
        ]
        
        # Print the table
        if self.verbose:
            print(tabulate(parameters, headers=["Parameter", "Value"], tablefmt="pretty"))
    
    def h(self, tau, constants):
       
        direction, xk = constants

        xk_expanded = xk.expand(self.int_samples_line, self.n_features)
        direction_expanded = direction.expand(self.int_samples_line, self.n_features)

        y = xk_expanded + tau * direction_expanded
       
        return self.f(y)
    
    def argmin_prox_line(self, tau_samples, constants):
       
        direction, xk = constants

        xk_expanded = xk.expand(self.int_samples_line, self.n_features)
        direction_expanded = direction.expand(self.int_samples_line, self.n_features)

        y = xk_expanded + tau_samples * direction_expanded

        prox_function = self.f(y) + 1/(2*self.t)*torch.norm(y-xk_expanded)**2

        prox_index = torch.argmin(prox_function)

        prox_tau = tau_samples[prox_index]
       
        return prox_tau
    
    def improve_prox_GH(self,xk, prox_xk, rescale_factor=1):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0#-torch.norm(xk - prox_xk)

        constants = direction, xk

        t = self.t

        delta = self.delta

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
          # Compute weights_line
          denominator = torch.dot(self.weights_line, exp_term)  # Scalar
          w = torch.where(
            denominator != 0,
            self.weights_line * exp_term / denominator,
            np.inf,#torch.full_like(self.weights, float("inf")),
          )

          # Check for Overflow
          softmax_overflow = 1.0 - (w < np.inf).prod()

          if softmax_overflow and rescale_factor > 1e-200:
              rescale_factor /= 2
              iterations+=1
          else:
              break

        grad_uk_L = (sigma/t_rescaled)*np.dot(w, self.z_line)


        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L

        prox_xk_new = xk + prox_tau*direction

        if self.plot:
            self.plot_1d_prox(xk.view(1, self.n_features).numpy(), prox_xk.numpy(), prox_xk_new.numpy(), t_rescaled)
        
        return prox_xk_new

    def compute_directional_prox_GH(self,xk):
        """
        Compute the prox of f using Gauss-Hermite Integration.
        
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

        iterations = 0

        # Reshape xk to ensure broadcasting
        # xk_expanded = xk.expand(self.int_samples_line, self.n_features)
        # z_expanded = self.z_line.expand(self.int_samples_line, self.n_features)
        # Reshape xk to ensure broadcasting
# Ensure xk has the correct shape for broadcasting (int_samples, n_features)
        xk_squeezed = xk.squeeze(0)  # Remove unnecessary dimensions (if xk has shape [1, 1, n_features])
        xk_expanded = xk_squeezed.expand(self.filtered_samples, self.n_features)  # Shape: (int_samples_line, n_features)

        # # Reshape z_line to allow broadcasting, then expand
        # z_expanded = self.z.unsqueeze(1)  # Shape: (int_samples, 1)
        # z_expanded = z_expanded.expand(self.int_samples, self.n_features)  # Shape: (int_samples_line, n_features)

        while True:
            # Apply Rescaling to time
            t_rescaled = t/rescale_factor

            sigma = np.sqrt(2*delta*t_rescaled)

            # Compute Perturbed Points
            y = xk_expanded - (sigma * self.z)

            # Compute Function Values
            f_values = self.f(y)  # Assume f supports batch processing, returns shape (int_samples,)
            rescaled_exponent = -rescale_factor * f_values / delta
            exp_term = torch.exp(rescaled_exponent)

            # print(f"{self.weights_line.shape=}")
            # print(f"{exp_term.shape=}")

            # Compute weights_line
            denominator = torch.dot(self.weights, exp_term)  # Scalar
            w = torch.where(
                denominator != 0,
                self.weights * exp_term / denominator,
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
        
        # Increase intial rescale factor if it is too small to better utilize samples
        # if iterations == 0:
        #     rescale_factor = min(scale_plus*rescale_factor, max_rescale)
        
        self.rescale0 = rescale_factor

        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()

        return prox_xk, rescale_factor
    
    def improve_prox_cauchy_argmin(self,xk, prox_xk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0

        constants = direction, xk

        # Adjust this parameter for heavier tails if needed
        gamma = self.t*self.delta

        # Generate Cauchy-distributed samples with location x_k and scale gamma
        cauchy_dist = torch.distributions.Cauchy(loc=tau_xk, scale=gamma)
        tau = cauchy_dist.sample((self.int_samples_line,)).view(-1, 1)

        prox_tau = self.argmin_prox_line(tau,constants)

        prox_xk_new = xk + prox_tau*direction
        
        return prox_xk_new
    
    def compute_directional_prox_argmin(self,xk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed
        gamma = self.t*self.delta

        # Expand xk to match the shape of y for broadcasting
        xk_expanded = xk.expand(self.int_samples, self.n_features)  # Shape: (self.int_samples, n_features)

        # Generate Cauchy-distributed samples with location x_k and scale gamma
        # Broadcasting ensures `loc=xk` (shape: (n_features,)) and `scale=gamma` (scalar)
        print(f"{self.distribution=}")
        if self.distribution == "Cauchy":
            cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=gamma)

            # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
            y = cauchy_dist.sample((self.int_samples,)) 
        else:
            y = xk + np.sqrt(gamma)*torch.randn(self.int_samples, self.n_features)


        # Compute the prox_function for each sample
        # f(y) must return a tensor of shape (self.int_samples,)
        prox_function = self.f(y)  + (1 / (2 * self.t)) * torch.norm(y - xk_expanded, dim=1)**2  # Shape: (self.int_samples,)

        # Find the index of the minimum value in prox_function
        prox_index = torch.argmin(prox_function) 

        # Use the index to extract the corresponding y value
        prox_xk = y[prox_index]#.unsqueeze(0)  # Shape: (1, n_features)

        return prox_xk
    
    def compute_directional_prox_softmax(self,xk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed
        delta = self.delta
        t = self.t*delta
        rescale_factor = 1
        iterations = 0

        xk = xk.squeeze(0)  # Remove unnecessary dimensions
        xk_expanded = xk.expand(self.int_samples, self.n_features) 

        while True:
            # Apply Rescaling to time
            t_rescaled = t/rescale_factor

            gamma = np.sqrt(delta*t_rescaled)

            # Compute Perturbed Points
            if self.distribution == "Cauchy":
                cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=gamma)

                # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
                y = cauchy_dist.sample((self.int_samples,))
            else:
                y = xk_expanded + gamma*torch.randn(self.int_samples, self.n_features)

            # Compute Function Values
            f_values = self.f(y)  
            rescaled_exponent = -rescale_factor * f_values / delta
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

        prox_function = self.f(y)  + (1 / (2 * self.t)) * torch.norm(y - xk_expanded, dim=1)**2  # Shape: (self.int_samples,)

        # Find the index of the minimum value in prox_function
        prox_index = torch.argmin(prox_function) 

        # Use the index to extract the corresponding y value
        print(f"Samples taken into account by softmax: {w[w>0.0].shape[0]}")
        print(f"argmin f(y): {self.f(y[prox_index].expand(1, self.n_features)).item()}")
        print(f"softmax f(y): {self.f(prox_xk).item()}")
        print(f"f(xk): {self.f(xk.expand(1, self.n_features)).item()}")
        print(f"delta: {delta}")

        if self.f(prox_xk).item() > self.f(xk.expand(1, self.n_features)).item():
            self.delta *= self.delta_dampener
            print(f"Decreasing delta: {delta}")

        return prox_xk

    
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

        # Initialize History
        xk_hist = torch.zeros(self.max_iters+1, self.n_features)
        xk_error_hist = torch.zeros(self.max_iters+1)
        fk_hist = torch.zeros(self.max_iters+1)

        f_k = self.f(xk.view(1, self.n_features))
        f_k_old = f_k
        error_k = torch.norm(xk - self.x_true)
 
        xk_hist[0, :] = xk
        xk_error_hist[0] = error_k
        fk_hist[0] = f_k

        # Define Outputs
        fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} '
        if self.verbose:
            print('-------------------------- RUNNING Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)
            print(fmt.format(0, fk_hist[0], xk_error_hist[0]))
        
        # Find Initial "Best Direction"
        if self.integration_method == "GH":
            prox_xk, rescale_factor  = self.compute_directional_prox_GH(xk)
        if self.integration_method == "argmin":
            prox_xk = self.compute_directional_prox_argmin(xk)
        else:
            prox_xk = self.compute_directional_prox_softmax(xk)

        f_prox = self.f(prox_xk.view(1, self.n_features))

        successful_ls_count = 0
        for k in range(self.max_iters):
            # Stopping Criteria
            if error_k < self.tol:
                if self.verbose:
                    print(f'HJ-MAD converged to tolerence {self.tol:6.2e}')
                    opt_error = torch.norm(xk - self.x_true)
                    print(f'iter = {k}, Error =  {opt_error:6.2e}')
                break
            # if k > 0:
            #     relative_gradient_error = np.abs(np.abs(torch.norm(xk_hist[k])/torch.norm(xk_hist[k-1]))-1)
            #     if relative_gradient_error < self.saturate_tol:
            #         print('HJ-MAD converged due to error saturation')
            #         print(f"Relative Gradient Error: {relative_gradient_error}")
            #         print(f"saturate_tol: {self.saturate_tol}")
            #         #self.distribution = "Normal"
            #         self.t *= 1.2



            # Find Proximal in 1D Along this Direction
            if self.int_samples_line > 0:
            #   if self.integration_method == "GH":
              prox_xk_new = self.improve_prox_GH(xk,prox_xk, rescale_factor=1)
            #   else:
            #     prox_xk_new = self.improve_prox_cauchy_argmin(xk,prox_xk)
              
              # Compute Function Value at New Proximal Point
              f_prox_new = self.f(prox_xk_new.view(1, self.n_features))

            #   print(f"{f_prox=}")
            #   print(f"Prox_xk error: {torch.norm(prox_xk - self.x_true):.2e}")
              print(f"{f_prox_new=}")
              print(f"Prox_xk_new error: {torch.norm(prox_xk_new - self.x_true):.2e}")
            #   print(f"{f_k=}")
            #   print(f"xk error: {torch.norm(xk - self.x_true):.2e}")

              # The function value with the full space proximal operator
              if f_prox_new < f_prox: # Update prox_xk if we have improvement
                print(f"Improvement from line search.\nR^n Prox_xk error: {torch.norm(prox_xk - self.x_true):.2e}, Line Prox_xk error: {torch.norm(prox_xk_new - self.x_true):.2e}")#, f_k: {f_k:.2e}, f_old: {f_prox:.2e}")
                successful_ls_count += 1
                prox_xk = prox_xk_new
                f_prox = f_prox_new
              else: # Don't update prox_xk if we don't have improvement
                print("No improvement in line search")

            # Only Update xk if we have improvement in function value
            xk = prox_xk
            f_k = f_prox # Update function value
            error_k = torch.norm(xk - self.x_true)

            # if f_prox < f_k_old:
            #     xk = prox_xk
            #     f_k = f_prox # Update function value
            #     error_k = torch.norm(xk - self.x_true)
            # else:
            #     f_k = f_k_old
            #     print("No improvement in function value")

            # Update History
            xk_hist[k+1, :] = xk
            xk_error_hist[k+1] = error_k
            fk_hist[k+1] = f_k
            f_k_old = f_k

            if self.verbose:
                print(fmt.format(k+1, fk_hist[k+1], xk_error_hist[k+1]))

            
            # Find Initial "Directional Prox in R^n"
            # Find Initial "Best Direction"
            if self.integration_method == "GH":
                prox_xk, rescale_factor  = self.compute_directional_prox_GH(xk.view(1, self.n_features))
            if self.integration_method == "argmin":
                prox_xk = self.compute_directional_prox_argmin(xk)
            else:
                prox_xk = self.compute_directional_prox_softmax(xk)
            f_prox = self.f(prox_xk.view(1, self.n_features))


        successful_ls_portion = successful_ls_count/(k)
        return xk, xk_hist[:k+1,:], xk_error_hist[:k+1], fk_hist[:k+1], successful_ls_portion
    
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

        plt.figure()
        plt.plot(tau_vals, f_vals, '-', color='black', label=f'f(x)')
        plt.plot(tau_vals, h_vals, '-', color='blue', label=r'$f(x) + \frac{1}{2t_0} ||x - x_0||^2$')
        plt.plot(tauk, self.f_numpy(xk[0,:]), '*', color='black', label=f'xk')
        plt.plot(tau_old, self.f_numpy(prox_xk_old[0,:]), '*', color='green', label=f'Full Space')
        plt.plot(tau_new, self.f_numpy(prox_xk_new[0,:]), '*', color='red', label=f'On Line')

        plt.xlabel(f'Tau')
        plt.ylabel('Function Value')
        plt.title(f'1D Descent for Chosen Line')
        plt.legend()
        plt.show()
