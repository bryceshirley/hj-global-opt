
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Optional 
from scipy.special import roots_hermite

from tabulate import tabulate


seed   = 30
torch.manual_seed(seed)

from hj_mad_cd import HJ_MAD


# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------

class Best_Direction_HJ_MAD:
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
    def __init__(self, f, f_numpy, f_name, x_true, delta=0.1, int_samples=1000,int_samples_line=100.0, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, alpha = 0.25,beta=0.9, theta=0.9, eta_vec = [0.9, 1.1], fixed_time=False, 
                 verbose=True,rescale0=1e-1,max_rescale= 1e3, momentum=0.0,saturate_tol=1e-9, integration_method='MC',plot=False, line_search=False):
      
        self.delta            = delta
        self.f                = f
        self.f_numpy          = f_numpy
        self.f_name           = f_name
        self.int_samples      = int_samples
        self.int_samples_line = int_samples_line
        self.max_iters        = max_iters
        self.tol              = tol
        self.alpha            = alpha
        self.beta             = beta
        self.t_vec            = t_vec
        self.theta            = theta
        self.x_true           = x_true
        self.eta_vec          = eta_vec
        self.fixed_time       = fixed_time
        self.verbose          = verbose
        self.momentum         = momentum
        self.rescale0         = rescale0
        self.saturate_tol     = saturate_tol
        self.integration_method = integration_method
        self.plot             = plot
        self.line_search      = line_search
        self.max_rescale      = max_rescale

        self.n_features = x_true.shape[0] 

        if self.line_search and self.integration_method == 'MC':
          self.weights = torch.ones(int_samples_line, dtype=torch.double)
        if self.line_search and self.integration_method == 'GH':
          z_line, weights = roots_hermite(int_samples_line)
          self.z_line = torch.tensor(z_line, dtype=torch.double)
          self.weights = torch.tensor(weights, dtype=torch.double)

        assert(alpha >= 1-np.sqrt(eta_vec[0]))
        assert(alpha <= 1+np.sqrt(eta_vec[1]))

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
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("t_init", format_scientific(self.t_vec[0])),
            ("t_min", format_scientific(self.t_vec[1])),
            ("t_max", format_scientific(self.t_vec[2])),
            ("theta", self.theta),
            #("x_true", self.x_true),
            ("eta_vec", self.eta_vec),
            ("fixed_time", self.fixed_time),
            ("momentum", self.momentum),
            ("rescale0", format_scientific(self.rescale0)),
            ("line_search", self.max_rescale),
            ("saturate_tol", format_scientific(self.saturate_tol)),
            ("integration_method", self.integration_method),
            ("line_search", self.line_search),
        ]
        
        # Print the table
        print(tabulate(parameters, headers=["Parameter", "Value"], tablefmt="pretty"))
    
    def h(self, tau, constants):
       
        direction, x0 = constants

        x0_expanded = x0.expand(self.int_samples_line, self.n_features)
        direction_expanded = direction.expand(self.int_samples_line, self.n_features)

        y = x0_expanded + tau * direction_expanded
       
        return self.f(y)
    
    def improve_prox(self,xk, t, rescale_factor, direction):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        
        tau_xk = 0
        #print(f"{direction=}")
        # underflow = 1e-15
        # overflow = 1e15
        # min_rescale=1e-10
        # iterations = 0
        constants = direction, xk

        # while True:
        # Apply Rescaling to time
        t_rescaled = t/rescale_factor

        sigma = np.sqrt(2*self.delta*t_rescaled)

        if self.integration_method == 'MC':
          self.z_line = torch.randn(self.int_samples_line,1)

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
        v_delta = torch.sum(self.weights * exp_term)

          # # Make sure over/underflow does not occur in the Denominator
          # if (v_delta >= underflow and v_delta <= overflow) or rescale_factor < min_rescale: 
          #   break
          
          # # Adjust rescale factor and increment iteration count
          # rescale_factor /= 2
          # iterations += 1

        # Compute Numerator Integral            
        numerator = torch.sum(self.weights * self.z_line * exp_term)

        # Compute Gradient in 1D
        grad_uk_L = (sigma/t_rescaled)*(numerator / v_delta)

        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L

        prox_xk = xk + prox_tau*direction

        grad_uk = grad_uk_L*direction  
        
        return t_rescaled*grad_uk, prox_xk

    def compute_directional_prox(self,x0, t):
      """
        Compute the prox of f using Monte Carlo Integration.
        
        Parameters:
            x (float): Point at which to compute the prox.
            t (float): Time parameter.
            delta (float): Viscousity parameter.
        Returns:
            prox (float): Computed prox.
        """
      
      underflow = 1e-15
      overflow = 1e15
      rescale_factor = self.rescale0
      scale_minus = 0.9
      scale_plus = 1.1
      min_rescale=1e-10
      max_rescale= self.max_rescale
      iterations = 0

      while True:
        # Sample From a Guassian with mean x and standard deviation
        standard_dev = np.sqrt(self.delta*t/rescale_factor)

        z = torch.randn(self.int_samples,self.n_features)

        y = standard_dev * z + x0

        f_values = self.f(y)

        # Rescale exponent to prevent overflow
        # print(f"f_values shape: {f_values.shape}")
        #print(f"rescale_factor: {rescale_factor}")
        # print(f"max_factor: {max_rescale}")
        # print(f"{max_rescale<rescale_factor}")
        # print(f"max_factor: {min_rescale}")
        # print(f"max f: {f_values.max()}")
        # print(f"self.delta: {self.delta}")
        rescaled_exponent = -rescale_factor*f_values/self.delta
        
        # Remove the maximum exponent to prevent overflow
        max_exponent = torch.max(rescaled_exponent) 
        shifted_exponent = rescaled_exponent - max_exponent

        exp_term = torch.exp(shifted_exponent)
        v_delta       = torch.mean(exp_term)

        print(f"  v_delta: {v_delta}")

        # Check for under/overflow
        if (v_delta >= underflow and v_delta <= overflow):
          break

        # Adjust rescale factor and increment iteration count
        rescale_factor = max(scale_minus*rescale_factor, min_rescale)
        iterations += 1
      
      # Increase intial rescale factor if it is too small to better utilize samples
      if iterations == 0:
        rescale_factor = min(scale_plus*rescale_factor, max_rescale)
      
      self.rescale0 = rescale_factor

      numerator = y*exp_term.view(self.int_samples, 1)
      numerator = torch.mean(numerator, dim=0)

      # Compute Estimated prox_xk
      prox_xk = numerator / (v_delta)

      grad_uk = (x0 -  numerator/(v_delta))

      print(f"  rescale_factor: {rescale_factor}")
      print(f"  iterations: {iterations}")

      return prox_xk, rescale_factor, grad_uk
    
    def find_direction(self,x0,t_init):

      # Execute the HJ_MAD_CD algorithm and retrieve results
      prox_x, rescale_factor, grad_uk = self.compute_directional_prox(x0, t_init)

      # Direction of line 1D.
      direction = (x0 - prox_x)/torch.norm(x0 - prox_x)

      return direction, rescale_factor, grad_uk, prox_x
    
    def update_time(self, tk, rel_grad_uk_norm):
      '''
        time step rule

        if ‖gk_plus‖≤ theta (‖gk‖+ eps):
          min (eta_plus t,T)
        else
          max (eta_minus t,t_min) otherwise

        OR:
        
        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)
      '''

      eta_minus = self.eta_vec[0]
      eta_plus = self.eta_vec[1]
      T = self.t_vec[2]
      t_min = self.t_vec[1]

      if rel_grad_uk_norm <= self.theta:
        # increase t when relative gradient norm is smaller than theta
        tk = min(eta_plus*tk , T) 
      else:
        # decrease otherwise t when relative gradient norm is smaller than theta
        tk = max(eta_minus*tk, t_min)

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
        xk = x0.clone()
        x_opt = xk.clone()
        if self.momentum is not None:
            xk_minus_1 = xk.clone()
        tk    = self.t_vec[0]
        fk_old = self.f(x_opt.view(1,-1))

        # Initialize History
        xk_hist = torch.zeros(self.max_iters+1, self.n_features)
        xk_error_hist = torch.zeros(self.max_iters+1)
        rel_grad_uk_norm_hist = torch.zeros(self.max_iters+1)
        fk_hist = torch.zeros(self.max_iters+1)
        tk_hist = torch.zeros(self.max_iters+1)

        rel_grad_uk_norm = 1.0

        # Find Initial "Best Direction"
        direction, rescale_factor, grad_uk, prox_xk  = self.find_direction(xk,tk)
        grad_uk_norm_old      = torch.norm(grad_uk)
        first_moment = (xk- prox_xk)

        # Define Outputs
        fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} '
        fmt += ' | |grad_uk| = {:6.2e} | tk = {:6.2e}'
        if self.verbose:
            print('-------------------------- RUNNING HJ-MAD ---------------------------')
            print('dimension = ', self.n_features, 'n_samples = ', self.int_samples)

        for k in range(self.max_iters):
            # Update History
            xk_hist[k, :] = xk
            rel_grad_uk_norm_hist[k] = rel_grad_uk_norm
            xk_error_hist[k] = torch.norm(xk - self.x_true)
            tk_hist[k] = tk
            fk_hist[k] = fk_old

            if self.verbose:
                print(fmt.format(k+1, fk_hist[k], xk_error_hist[k], rel_grad_uk_norm_hist[k], tk))

            # Find Proximal in 1D Along this Direction
            if self.line_search:
              grad_uk, prox_xk_new = self.improve_prox(xk,tk,rescale_factor,direction)
              f_prox_new = self.f(prox_xk_new.view(1, self.n_features))
              f_prox = self.f(prox_xk.view(1, self.n_features))
              if f_prox_new < f_prox:
                prox_xk = prox_xk_new
                print("Improvement from line search")
              else:
                print("No improvement in line search")

            # Accelerate Gradient Descent if momentum is not none
            if k > 0 and self.momentum != 0.0:  # when k=0 go to else with as x_{-1} = x_0
                yk = xk.clone()
                yk = xk + self.momentum * (xk - xk_minus_1)
                xk_minus_1 = xk
            else:
                yk = xk.clone()

            # Perform gradient descent update
            first_moment  = self.beta*first_moment + (1-self.beta)*(yk - prox_xk)

            xk = yk - self.alpha *(yk-prox_xk)

            if self.plot:
              self.plot_1d_prox(xk.numpy(), prox_xk.numpy(), tk, direction.numpy())
            
            # Find New "Best Direction"
            if self.line_search:
               direction, rescale_factor, _, prox_xk  = self.find_direction(xk,tk)
            else:
               direction, rescale_factor, grad_uk, prox_xk  = self.find_direction(xk,tk)

            # Only Update x_opt if the new point is better in f
            fk = self.f(xk.view(1, self.n_features))
            if fk_old < fk:
                x_opt = xk.clone()
            fk_old = fk

            # Stopping Criteria
            if xk_error_hist[k] < self.tol:
                if self.verbose:
                  print('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
                  print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
                break
            if k > 0 and np.abs(torch.norm(xk_hist[k] - xk_hist[k-1])) < self.saturate_tol*torch.norm(xk_hist[k-1]): 
                if self.verbose:
                  print('HJ-MAD converged due to error saturation with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
                  print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
                    #self.rescale0 = 1
                break
            
            # Update time
            grad_uk_norm      = torch.norm(grad_uk)
            rel_grad_uk_norm  = grad_uk_norm / (grad_uk_norm_old + 1e-12)
            grad_uk_norm_old  = grad_uk_norm

            if self.fixed_time == False:
                tk = self.update_time(tk, rel_grad_uk_norm)
                

        return x_opt, xk_hist[:k+1,:], xk_error_hist[:k+1], rel_grad_uk_norm_hist[:k+1], fk_hist[:k+1], tk_hist[:k+1]
    
    def plot_1d_prox(self, x0, xk, tk, direction, num_points=1000):
        """
        Plots the 1D descent for the current dimension.

        Args:
            xk (torch.Tensor): Current position.
            dim (int): The current dimension being optimized.
            domain (tuple): The range over which to vary the current dimension.
            num_points (int): Number of points to sample in the domain.
        """

        tau0 = 0
        tauk = np.dot(xk - x0, direction)
        f_vals = []
        h_vals = []

        tau_vals = np.linspace(-5*tauk, 5*tauk, num_points)
        # Create x_vals with shape (num_points, 2)
        x_vals = np.zeros((num_points, self.n_features))

        # Compute x_vals by adding tau * direction for each component of x0
        for i in range(self.n_features):
            x_vals[:, i] = x0[i] + tau_vals * direction[i]


        for x in x_vals:
            xk_varied = x
            f_vals.append(self.f_numpy(xk_varied))
            h_vals.append(self.f_numpy(xk_varied) + 1/(2*tk)*np.linalg.norm(xk_varied-x0)**2)


        # print(f"{self.delta} * {self.t_vec[0]} = {self.delta * self.t_vec[0]}")
        # print(f'Std Dev: {std_dev}, Std Dev Minus: {std_dev_minus}, Std Dev Plus: {std_dev_plus}')

        plt.figure()
        plt.plot(tau_vals, f_vals, '-', color='black', label=f'f(x)')
        plt.plot(tau_vals, h_vals, '-', color='blue', label=r'$f(x) + \frac{1}{2t_0} ||x - x_0||^2$')
        plt.plot(tau0, self.f_numpy(x0), '*', color='red', label=f'tau at x0')
        plt.plot(tauk, self.f_numpy(xk), '*', color='green', label=f'tau at xk')

        plt.xlabel(f'Tau')
        plt.ylabel('Function Value')
        plt.title(f'1D Descent for Chosen Line')
        plt.legend()
        plt.show()
    