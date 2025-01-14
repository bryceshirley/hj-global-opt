
import numpy as np
import torch

from scipy.special import roots_hermite

from tabulate import tabulate

import matplotlib.pyplot as plt
from itertools import product

seed   = 30
torch.manual_seed(seed)


from typing import Callable, Literal, Optional, Tuple


class AdaptiveDeltaConfig:
    """
    Configuration for adaptive delta updates.

    Attributes:
        delta_minus (float): Factor to decrease delta (default: 0.99).
        delta_plus (float): Factor to increase delta (default: 1.01).
        saturate_tol (float): Tolerance level for saturating updates (default: 1e-4).
    """
    def __init__(self, delta_minus: float = 0.99, delta_plus: float = 1.01, saturate_tol: float = 1e-4) -> None:
        self.saturate_tol = saturate_tol
        self.delta_minus = delta_minus
        self.delta_plus = delta_plus


class AdaptiveTimeConfig:
    """
    Configuration for adaptive time step updates.

    Attributes:
        t_min (float): Minimum allowable time step (default: 1e-2).
        t_max (float): Maximum allowable time step (default: 1e2).
        theta (float): parameter used to update tk (default: 0.9).
        eta_plus (float): Factor to increase time step (default: 1.01).
        eta_minus (float): Factor to decrease time step (default: 0.99).
    """
    def __init__(self, t_min: float = 1e-2, t_max: float = 1e2, theta: float = 0.9, eta_plus: float = 1.01, eta_minus: float = 0.99) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.theta = theta
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus


class PlotConfig:
    """
    Configuration for plotting.

    Attributes:
        f_numpy (Callable): The numpy-compatible version of the function to be minimized.
        f_name (str): The name of the function, used for labeling plots.
    """
    def __init__(self, f_numpy: Callable, f_name: str) -> None:
        self.f_numpy = f_numpy
        self.f_name = f_name

    def plot_1d_prox(self, xk: float, prox_xk_old: float, prox_xk_new: float, tk: float, num_points=1000):
        """
        Plots the 1D descent for the current dimension.
        """

        tauk = 0
        direction = (xk - prox_xk_old)/np.linalg.norm(xk - prox_xk_old)
        n_features = xk.shape[0]

        tau_old = np.dot(prox_xk_old - xk, direction.T)
        tau_new = np.dot(prox_xk_new - xk, direction.T)
        f_vals = []
        h_vals = []

        tau_vals = np.linspace(-40, 20, num_points)
        # Create x_vals with shape (num_points, 2)
        x_vals = np.zeros((num_points, n_features))

        # Compute x_vals by adding tau * direction for each component of x0
        for i in range(n_features):
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


class HJ_MAD_LS:
    """
    Hamilton-Jacobi Moreau Adaptive Descent with Line Search (HJ_MAD_LS).

    This algorithm solves nonconvex minimization problems using a zeroth-order sampling approach 
    and incorporates line search for step size optimization.

    Attributes:
        f (Callable): Function to be minimized, accepting inputs of shape (n_samples x n_features).
        x_true (float): True global minimizer for comparison and validation.
        delta (float): Coefficient for the viscous term in the Hamilton-Jacobi equation.
        t (float): Initial time step for the optimization process.
        int_samples (int): Number of samples used to approximate expectations.
        max_iters (Union[int, float]): Maximum number of iterations (default: 5e4).
        tol (float): Stopping tolerance (default: 5e-2).
        adaptive_delta_config (Optional[AdaptiveDeltaConfig]): Configuration for adaptive delta updates.
        adaptive_time_config (Optional[AdaptiveTimeConfig]): Configuration for adaptive time updates.
        plot_config (Optional[PlotConfig]): Configuration for plotting function values.
        integration_method (Literal["MC", "GH"]): Integration method, either Monte Carlo ("MC") or Gauss-Hermite ("GH").
        distribution (Literal["Gaussian", "Cauchy"]): Sampling distribution, either Gaussian or Cauchy.
        line_search (bool): Enables line search for step size optimization (default: True).
        step_size (float): Initial step size (ignored if line search is enabled).
        beta (float): Momentum factor for exponential averaging of gradients.
        momentum (float): Additional momentum parameter for acceleration.
        verbose (bool): Prints parameter information and updates if True.

    Outputs:
        x_opt (np.ndarray): Approximation of the optimal solution.
        xk_hist (np.ndarray): History of solutions during optimization.
        xk_error_hist (np.ndarray): History of errors during optimization.
        fk_hist (list): History of function values during optimization.
        deltak_hist (np.ndarray): History of delta values during optimization.
        tk_hist (np.ndarray): History of time step values during optimization.
        successful_ls_portion (float): Proportion of successful line search steps.
    """
    def __init__(
        self,
        f: Callable,
        x_true: float,
        delta: float = 0.1,
        t: float = 1e2,
        int_samples: int = 150,
        max_iters: int = int(5e4),
        tol: float = 5e-2,
        adaptive_delta_config: Optional[AdaptiveDeltaConfig] = None,
        adaptive_time_config: Optional[AdaptiveTimeConfig] = None, 
        plot_config: Optional[PlotConfig] = None,
        integration_method: Literal["MC", "GH"] = "MC",
        distribution: Literal["Gaussian", "Cauchy"] = "Gaussian",
        line_search: bool = True,
        step_size: float = 1.0,
        beta: float = 0.0,
        momentum: float = 0.0,
        verbose: bool = True,
        eps0: float = 0.1,
        cooling: float = 0.1
    ) -> None:
      
        # General configurations
        self.f = f
        self.x_true = x_true
        self.delta = delta
        self.t = t
        self.verbose = verbose
        self.max_iters = max_iters
        self.tol = tol
        self.integration_method = integration_method
        self.distribution = distribution
        self.int_samples = int_samples
        self.beta = beta
        self.momentum = momentum
        self.n_features = x_true.shape[0]
        self.eps0 = eps0
        self.cooling = cooling

        # Adaptive delta configurations
        self.adaptive_delta = adaptive_delta_config is not None
        if self.adaptive_delta:
            self.delta_plus = adaptive_delta_config.delta_plus
            self.delta_minus = adaptive_delta_config.delta_minus
            self.saturate_tol = adaptive_delta_config.saturate_tol

        # Adaptive time configurations
        self.adaptive_time = adaptive_time_config is not None
        if self.adaptive_time:
            self.t_min = adaptive_time_config.t_min
            self.t_max = adaptive_time_config.t_max
            self.theta = adaptive_time_config.theta
            self.eta_plus = adaptive_time_config.eta_plus
            self.eta_minus = adaptive_time_config.eta_minus

        # Validate step size for adaptive time
        if self.adaptive_time:
            assert 1 - np.sqrt(self.eta_minus) <= step_size <= 1 + np.sqrt(self.eta_plus)
        self.step_size = step_size

        # Line search configuration
        self.line_search = line_search
        if self.line_search:
            z_line, weights_line = roots_hermite(self.int_samples)
            self.z_line = torch.tensor(z_line, dtype=torch.double)
            self.weights_line = torch.tensor(weights_line, dtype=torch.double)
            self.step_size = 1.0

        # Plotting configuration
        self.plot_config = plot_config

        # Gauss-Hermite grid for integration
        if integration_method == "GH":
            self.weights, self.z, self.filtered_samples = self.generate_gh_grid_matrix_with_threshold()



    def generate_gh_grid_matrix_with_threshold(self):
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

        if self.adaptive_delta:
            eps = -(1-deltak)
        else:
            eps = self.eps0


        iterations=0
        rescale_factor=1
        while True:
          # Apply Rescaling to time
          t_rescaled = tk/rescale_factor

          sigma = np.sqrt(2*t_rescaled*(1+eps))

          # Compute Function Values
          tau = tau_xk - self.z_line*sigma # Size (int_samples,1)
          z = xk_expanded + tau.view(-1, 1) * direction_expanded

          f_values = self.f(z) + (1/(2*t_rescaled))*(eps+(1-deltak))*(torch.norm(z, p=2, dim=1)**2) 

          # Apply Rescaling to Exponent
          rescaled_exponent = - rescale_factor*f_values/ deltak

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

        if self.plot_config is not None:
            self.plot_config.plot_1d_prox(xk.view(1, self.n_features).numpy(), prox_xk.numpy(), prox_xk_new.numpy(), t_rescaled)
        
        return prox_xk_new

    def compute_directional_prox_gauss_hermite(self,xk,deltak,tk):
        """
        Compute the prox of f using Gauss-Hermite Integration.
        
        Parameters:
            x (float): Point at which to compute the prox.
            t (float): Time parameter.
            delta (float): Viscousity parameter.
        Returns:
            prox (float): Computed prox.
        """

        rescale_factor = 1.0
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
            t_rescaled = tk/rescale_factor

            sigma = np.sqrt(2*deltak*t_rescaled)

            # Compute Perturbed Points
            y = xk_expanded - (sigma * self.z)

            # Compute Function Values
            f_values = self.f(y)  # Assume f supports batch processing, returns shape (int_samples,)
            rescaled_exponent = -rescale_factor * f_values / deltak
            rescaled_exponent = rescaled_exponent - torch.max(rescaled_exponent)
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
    

        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()

        return prox_xk
    
    # def compute_directional_prox(self,xk,deltak,tk):
    #     '''
    #         Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
    #         Corresponding to the Proximal Operator.
    #     '''
    #     # Adjust this parameter for heavier tails if needed

    #     rescale_factor = 1
    #     iterations = 0

    #     xk = xk.squeeze(0)  # Remove unnecessary dimensions
    #     xk_expanded = xk.expand(self.int_samples, self.n_features) 

    #     while True:
    #         # Apply Rescaling to time
    #         t_rescaled = tk/rescale_factor

    #         standard_deviation = np.sqrt(deltak*t_rescaled)

    #         # Compute Perturbed Points
    #         if self.distribution == "Cauchy":
    #             cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=standard_deviation)

    #             # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
    #             y = cauchy_dist.sample((self.int_samples,))
    #         else:
    #             y = xk_expanded + standard_deviation*torch.randn(self.int_samples, self.n_features)

    #         # Compute Function Values
    #         f_values = self.f(y)  
    #         rescaled_exponent = -rescale_factor * f_values / deltak
    #         rescaled_exponent = rescaled_exponent - torch.max(rescaled_exponent)
    #         exp_term = torch.exp(rescaled_exponent)

    #         # Compute weights_line
    #         denominator = torch.sum(exp_term)  # Scalar
    #         w = torch.where(
    #             denominator != 0,
    #             exp_term / denominator,
    #             np.inf,#torch.full_like(self.weights, float("inf")),
    #         )

    #         # Check for Overflow
    #         softmax_overflow = 1.0 - (w < np.inf).prod()

    #         if softmax_overflow and rescale_factor > 1e-200:
    #             # Adjust rescale factor and increment iteration count
    #             rescale_factor /= 2
    #             iterations += 1
    #         else:
    #             break

    #     prox_xk = torch.matmul(w.t(), y)
    #     prox_xk = prox_xk.view(-1,1).t()
          
    #     l2_norm_squared = torch.norm(xk - prox_xk, p=2, dim=1)**2

    #     f_prox_xk = self.f(prox_xk.view(1, self.n_features))

    #     f_regularized_prox = self.f(prox_xk.view(1, self.n_features)) + (l2_norm_squared/ (2 * tk))

    #     print(f"    {f_regularized_prox.item()=},{f_prox_xk.item()=}")
    #     print(f"    {torch.max(w).item()=}")


    #     return prox_xk
    
    def compute_directional_prox(self,xk,deltak,tk):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed

        xk = xk.squeeze(0)  # Remove unnecessary dimensions
        xk_expanded = xk.expand(self.int_samples, self.n_features) 

        if self.adaptive_delta or self.adaptive_time:
            if self.distribution == "Cauchy":
                eps = np.sqrt(2*deltak/tk)-1
            else:
                eps = deltak - 1
        else:
            eps = self.eps0

        # Compute Perturbed Points
        if self.distribution == "Cauchy":
            scale = tk*(1+eps)

            cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=scale)

            # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
            y = cauchy_dist.sample((self.int_samples,))

            l2_norm_squared = torch.norm(xk_expanded - y, p=2, dim=1)**2

            exponent = -(1/deltak)*(self.f(y) + (l2_norm_squared/ (2 * tk)) - deltak*torch.log((l2_norm_squared/(scale**2)) + 1))
            rescaled_exponent = exponent - torch.max(exponent)
        else:
            standard_deviation = np.sqrt(tk*(1+eps))

            y = xk_expanded + standard_deviation*torch.randn(self.int_samples, self.n_features)

            l2_norm_squared = torch.norm(xk_expanded - y, p=2, dim=1)**2

            exponent = -(1/deltak)*((self.f(y)) + (1-deltak/(1+eps))*l2_norm_squared/ (2 * tk))

        # Compute Function Values
        rescaled_exponent = exponent - torch.max(exponent)
        exp_term = torch.exp(rescaled_exponent)

        # Compute weights_line
        denominator = torch.sum(exp_term)  # Scalar
        w = torch.where(
            denominator != 0,
            exp_term / denominator,
            np.inf,#torch.full_like(self.weights, float("inf")),
        )


        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()

        return prox_xk
    
    def update_time(self, tk, rel_grad_uk_norm):
      '''
        time step rule

        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)
      '''

      if rel_grad_uk_norm <= self.theta:
        # Decrease t when relative gradient norm is smaller than theta
        tk = max(self.eta_minus*tk, self.t_min)
      else:
        # Increas otherwise t when relative gradient norm is smaller than theta
        tk = min(self.eta_plus*tk, self.t_max)

      return tk

    def run(self, x0: torch.Tensor):
        """
        Runs the coordinate descent optimization process using the HJ-MAD-LS algorithm.

        Args:
            x0 (torch.Tensor): Initial guess for the minimizer, with shape `(n_features,)`.

        Returns:
            - torch.Tensor: xk - The optimal solution found by the optimization process, with shape `(n_features,)`.
            - torch.Tensor: xk_hist - History of solutions for each iteration, with shape `(n_features, num_iterations)`.
            - torch.Tensor: xk_error_hist - Error history for each iteration, with shape `(num_iterations,)`.
            - torch.Tensor: fk_hist - Objective function values for each iteration, with shape `(num_iterations,)`.
            - torch.Tensor: deltak_hist - History of delta values for each iteration, with shape `(num_iterations,)`.
            - torch.Tensor: tk_hist - History of time parameter `t` for each iteration, with shape `(num_iterations,)`.
            - float: successful_ls_portion - The proportion of iterations where line search improved the result.

        """

        # Initialize Variables
        xk = x0.clone()
        xk_minus_1 = x0.clone()
        tk = self.t
        deltak = self.delta

        # Initialize History
        self.n_features = x0.shape[0]
        fk_hist = torch.zeros(self.max_iters+1)
        xk_hist = torch.zeros(self.n_features,self.max_iters+1)
        deltak_hist = torch.zeros(self.max_iters+1)
        xk_error_hist = torch.zeros(self.max_iters+1)
        tk_hist = torch.zeros(self.max_iters+1)
        fk = self.f(xk.view(1, self.n_features))
        xk_hist[:,0] = xk
        fk_hist[0] = fk
        deltak_hist[0] = deltak
        tk_hist[0] = tk
        xk_error_hist[0] = torch.norm(xk - self.x_true)

        # Define Outputs
        fmt = '[{:3d}]: fk = {:6.2e} | error = {:6.2e} | deltak = {:6.2e} | tk = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD-LS Algorithm ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)
            print(fmt.format(0,xk_error_hist[0], fk_hist[0], deltak, tk))

        saturation_count = 0
        successful_ls_portion=0

        for k in range(self.max_iters):
            # Compute Proximal Point
            if self.integration_method == "GH":
                prox_xk  = self.compute_directional_prox_gauss_hermite(xk,deltak,tk)
            else:
                prox_xk = self.compute_directional_prox(xk,deltak,tk)

            f_prox = self.f(prox_xk.view(1, self.n_features))

            # Update Delta
            if self.adaptive_delta:
                # Dampen delta if the proximal point is worse than the current point
                if f_prox >= fk: 
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
                if f_prox_new < f_prox:
                    prox_xk = prox_xk_new
                    f_prox = f_prox_new
                    successful_ls_portion += 1
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
            xk = yk - self.step_size*first_moment
            fk = self.f(xk.view(1, self.n_features))
            errork = torch.norm(xk - self.x_true)
            
            # Update time
            if self.adaptive_time:
                grad_norm = torch.norm(first_moment)
                if k >0:
                    rel_grad_norm = grad_norm/(grad_norm_old + 1e-12)
                    tk = self.update_time(tk,rel_grad_norm)
                grad_norm_old = grad_norm

            # Print Iteration Information
            if self.verbose:
                print(fmt.format(k+1, fk.item(),errork, deltak, tk))

            # Update History
            xk_hist[:,k+1] = xk
            xk_error_hist[k+1] = errork.item()
            fk_hist[k+1] = fk
            deltak_hist[k+1] = deltak
            tk_hist[k+1] = tk

            # Stopping Criteria
            if errork < self.tol: # f value is less than tolerance
                if self.verbose:
                    print(f'    HJ-MAD converged to tolerence {self.tol:6.2e}')
                    print(f'    iter = {k}: f_value =  {fk}')
                break

        
        if k == self.max_iters-1:
            if self.verbose:
                print(f"    HJ-MAD did not converge after {self.max_iters} iterations")
    
        return xk, xk_hist[:k+2,:], xk_error_hist[:k+2], fk_hist[:k+2], deltak_hist[:k+2], tk_hist[:k+2], successful_ls_portion/(k+2)
    
    def HJ_simulated_annealing(self, x0):
        # self.f = f
        xk = x0.clone()
        delta0 = self.delta
        xk_minus_1 = x0.clone()
        cooling_rate = self.cooling

        # Initialize History
        self.n_features = x0.shape[0]
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
                deltak = delta0/(1+cooling_rate*((tk**3)))
                #deltak = delta0/np.log(tk+np.e) # Classic Cooling

                # Approximate Brownian / Boltzmann Expectation
                prox_xk = self.compute_directional_prox(xk,deltak,tk)# torch.distributions.Cauchy(loc=xk, scale=scale).sample()

                if self.line_search and tk > 0:
                    prox_xk = self.improve_prox_with_line_search(xk, prox_xk,deltak,tk)

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
                xk = yk - self.step_size*first_moment
                fk = self.f(xk.view(1, self.n_features))

                if fk <fk_opt:
                    fk_opt = fk
                    xk_opt = xk

                # Print Iteration Information
                if tk % 10 == 1:
                    if self.verbose:
                        print(fmt.format(tk, fk.item(), deltak, tk,deltak*tk))
                # Update History
                xk_hist[:,tk+1] = xk
                fk_hist[tk+1] = fk
                deltak_hist[tk+1] = deltak
                p_hist[tk+1] = p

                # Stopping Criteria
                if fk < self.tol: # f value is less than tolerance
                    if self.verbose:
                        print(f'    HJ-MAD converged to tolerence {self.tol:6.2e}')
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
            if fk < self.tol: 
                break  
        
        return xk_opt, xk_hist[:,2:tk+1], fk_hist[2:tk+1], deltak_hist[2:tk+1],p_hist[2:tk+1]

#x_opt_cauchy, xk_hist_cauchy, fk_hist_cauchy, delta_hist_cauchy, tk_hist_cauchy