
import numpy as np
import torch

from scipy.special import roots_hermite

seed   = 30
torch.manual_seed(seed)


class HJ_PROX_LS:

    def __init__(self, delta=100, t = 1e1, int_samples=100, distribution="Gaussian",delta_dampener=0.99,beta=0.5,verbose=True):
        # Algorithm Parameters
        self.delta            = delta
        self.t                = t
        self.int_samples      = int_samples

        self.delta_dampener =delta_dampener
        self.beta = beta
        self.distribution = distribution

        self.verbose          = verbose

        # Generate Hermite Quadrature Points in R^1
        z_line, weights_line = roots_hermite(int_samples)
        self.z_line = torch.tensor(z_line, dtype=torch.double)
        self.weights_line = torch.tensor(weights_line, dtype=torch.double)

    def improve_prox_with_line_search(self,xk, prox_xk,f):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Direction of line 1D.
        direction = (xk - prox_xk)/torch.norm(xk - prox_xk)
        tau_xk = 0#-torch.norm(xk - prox_xk)

        # Expand xk and direction
        xk_expanded = xk.expand(self.int_samples, self.n_features)
        direction_expanded = direction.expand(self.int_samples, self.n_features)
        

        rescale_factor=1
        while True:
          # Apply Rescaling to time
          t_rescaled = self.t/rescale_factor

          sigma = np.sqrt(2*self.delta*t_rescaled)

          # Compute Function Values
          tau = tau_xk - self.z_line*sigma # Size (int_samples,1)

          y = xk_expanded + tau.view(-1, 1) * direction_expanded

          f_values = f(y) # Size (int_samples,1)

          # Apply Rescaling to Exponent
          rescaled_exponent = - rescale_factor*f_values/ self.delta

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
    
    def compute_prox(self,xk,f):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        # Adjust this parameter for heavier tails if needed
        rescale_factor = 1

        xk = xk.squeeze(0)  # Remove unnecessary dimensions
        xk_expanded = xk.expand(self.int_samples, self.n_features) 

        while True:
            # Apply Rescaling to time
            t_rescaled = self.t/rescale_factor

            standard_deviation = np.sqrt(self.delta*t_rescaled)

            # Compute Perturbed Points
            if self.distribution == "Cauchy":
                cauchy_dist = torch.distributions.Cauchy(loc=xk, scale=standard_deviation)

                # Sample `self.int_samples` points, result shape: (self.int_samples, n_features)
                y = cauchy_dist.sample((self.int_samples,))
            else:
                y = xk_expanded + standard_deviation*torch.randn(self.int_samples, self.n_features)

            # Compute Function Values
            f_values = f(y)  
            rescaled_exponent = -rescale_factor * f_values / self.delta
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
            else:
                break

        prox_xk = torch.matmul(w.t(), y)
        prox_xk = prox_xk.view(-1,1).t()
        f_prox = f(prox_xk).item()

        # Improve the proximal point using line search
        prox_xk_new = self.improve_prox_with_line_search(xk,prox_xk,f)
        f_prox_new = f(prox_xk_new.view(1, self.n_features))

        # Check if the line search improved the proximal point
        if f_prox_new < f_prox:
            prox_xk = prox_xk_new
        # else:
        #     if self.verbose:
        #         print("No improvement from line search.")

        # Return the proximal point for xk
        return prox_xk

    def run(self, f, x, first_moment=None):
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
        self.n_features = x.shape[1]
        fx = f(x.view(1, self.n_features))

        # Compute Proximal Point and Function Value
        prox = self.compute_prox(x,f)
        fprox = f(prox.view(1, self.n_features))

        # Update Delta
        if fprox < fx:
            self.delta = self.delta * 1.01
            if self.verbose:
                print(f'Loss decreased: {fx} -> {fprox}. Delta: {self.delta}')
        else:
            self.delta = self.delta * self.delta_dampener
            if self.verbose:
                print(f'Loss increased: {fx} -> {fprox}. Delta: {self.delta}')

        # Update x
        if first_moment is None or self.beta == 0.0:
            first_moment = prox
        first_moment = self.beta*first_moment+(1-self.beta)*(x-prox)
        x_new = x - first_moment
        fx_new = f(x_new.view(1, self.n_features))

        return x_new, fx_new, first_moment
