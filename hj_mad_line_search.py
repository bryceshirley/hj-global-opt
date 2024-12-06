
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Optional 
from scipy.special import roots_hermite


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
    def __init__(self, f, f_numpy,x_true, delta=0.1, int_samples=100, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, theta=0.9, beta=[0.9], eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, 
                 verbose=True,rescale0=1e-1, momentum=None,saturate_tol=1e-9, integration_method='MC',plot=False):
      
        self.delta            = delta
        self.f                = f
        self.f_numpy          = f_numpy
        self.int_samples      = int_samples
        self.max_iters        = max_iters
        self.tol              = tol
        self.t_vec            = t_vec
        self.theta            = theta
        self.x_true           = x_true
        self.beta             = beta 
        self.alpha            = alpha 
        self.eta_vec          = eta_vec
        self.fixed_time       = fixed_time
        self.verbose          = verbose
        self.momentum         = momentum
        self.rescale0         = rescale0
        self.saturate_tol     = saturate_tol
        self.integration_method = integration_method
        self.plot             = plot

        self.n_features = x_true.shape[0]

        if self.integration_method == 'MC':
          self.weights = torch.ones(self.int_samples, dtype=torch.double)
          self.z = torch.randn(self.int_samples,1)
        else:
          z, weights = roots_hermite(int_samples)
          self.z = torch.tensor(z, dtype=torch.double)
          self.weights = torch.tensor(weights, dtype=torch.double)


        
        # check that alpha is in right interval
        assert(alpha >= 1-np.sqrt(eta_vec[0]))
        assert(alpha <= 1+np.sqrt(eta_vec[1]))
    
    def h(self, tau, constants):
       
        direction, x0 = constants

        x0_expanded = x0.expand(self.int_samples, self.n_features)
        direction_expanded = direction.expand(self.int_samples, self.n_features)

        y = x0_expanded + tau * direction_expanded
       
        return self.f(y)
    
    def compute_prox_tau(self,x0, xk, t, rescale_factor, direction):
        '''
            Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
            Corresponding to the Proximal Operator.
        '''
        
        tau_xk = torch.dot(x0-xk, direction)
        #print(f"{direction=}")
        constants = direction, x0

        # Apply Rescaling to time
        t_rescaled = t/rescale_factor

        sigma = np.sqrt(2*self.delta*t_rescaled)



        # Compute Function Values
        tau = tau_xk - self.z*sigma # Size (int_samples,1)
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

        # Compute Numerator Integral            
        numerator = torch.sum(self.weights * self.z * exp_term)

        # Compute Gradient in 1D
        grad_uk_L = (sigma/t_rescaled)*(numerator / v_delta)

        # Compute Prox_xk
        prox_tau = tau_xk - t_rescaled*grad_uk_L

        prox_xk = x0 + prox_tau*direction

        grad_uk = grad_uk_L*direction  

        self.plot_1d_prox(x0.numpy(), prox_xk.numpy(), t, direction.numpy())
        
        return -t_rescaled*grad_uk

    # def check_same_side(self,x0, x1, x2):
    #     # Compute direction vectors
    #     v1 = x0 - x1
    #     v2 = x0 - x2
        
    #     # Compute dot product
    #     dot_product = np.dot(v1, v2)
        
    #     # Determine side
    #     if dot_product >= 0:
    #         return True  # Same side
    #     elif dot_product < 0:
    #         return False  # Opposite sides

    # def compute_prox_tau_MC(self,x0, prox_x0, xk, t, direction, initial_rescale_factor: Optional[float]=None, min_rescale=1e-10):
    #   '''
    #     Rescale the Exponent For Under/OverFlow and find the line parameter Tau 
    #     Corresponding to the Proximal Operator.
    #   '''
    #   if initial_rescale_factor is None:
    #     initial_rescale_factor = self.rescale0
        
    #   rescale_factor = initial_rescale_factor
    #   iterations = 0

    #   int_samples=self.int_samples

    #   constants = direction, x0


    #   while True:
    #     standard_dev = np.sqrt(self.delta*t/rescale_factor)


    #     if self.check_same_side(x0, prox_x0, xk):
    #         tau_mean = torch.norm(x0 - xk)
    #     else:
    #         tau_mean = -torch.norm(x0 - xk)

    #     tau = standard_dev * torch.randn(int_samples,1) +  tau_mean# tau is sized (n_samples,1)

    #     # Sample points along line and compute function values
    #     h_values = self.h(tau,constants)

    #     rescaled_exponent = -rescale_factor*h_values/self.delta
    #     max_exponent = torch.max(rescaled_exponent)  # Find the maximum exponent
    #     shifted_exponent = rescaled_exponent - max_exponent
    #     exp_term = torch.exp(shifted_exponent)
    #     v_delta       = torch.sum(exp_term)

    #     if (v_delta >= 1e-15 and v_delta <= 1e15) or rescale_factor < min_rescale:
    #       break

    #     # Adjust rescale factor and increment iteration count
    #     rescale_factor /= 2
    #     iterations += 1

    #   numerator = tau*exp_term.view(self.int_samples, 1)
    #   numerator = torch.sum(numerator, dim=0)

    #   prox_tau = numerator/v_delta

    #   grad_uk = -(rescale_factor/t)*prox_tau*direction
        
    #   return prox_tau, grad_uk
    
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
      rescale_factor = 1
      min_rescale=1e-10
      iterations = 0
      int_samples = int(1e8)
      z = torch.randn(int_samples,self.n_features) 

      while True:
        # Sample From a Guassian with mean x and standard deviation
        standard_dev = np.sqrt(self.delta*t/rescale_factor)
        y = standard_dev * z + x0

        f_values = self.f(y)

        # Rescale exponent to prevent overflow
        rescaled_exponent = -rescale_factor*f_values/self.delta

        
        # Remove the maximum exponent to prevent overflow
        max_exponent = torch.max(rescaled_exponent) 
        shifted_exponent = rescaled_exponent - max_exponent

        exp_term = torch.exp(shifted_exponent)
        v_delta       = torch.mean(exp_term)

        # Check for under/overflow
        if (v_delta >= underflow and v_delta <= overflow) or rescale_factor < min_rescale:
          break

        # Adjust rescale factor and increment iteration count
        rescale_factor /= 2
        iterations += 1

      numerator = y*exp_term.view(int_samples, 1)
      numerator = torch.mean(numerator, dim=0)

      # Compute Estimated prox_xk
      prox_xk = numerator / (v_delta)

      return prox_xk, rescale_factor
    
    def find_direction(self,x0,t_init):

      # Execute the HJ_MAD_CD algorithm and retrieve results
      prox_x, rescale_factor = self.compute_directional_prox(x0, t_init)

      # Direction of line 1D.
      direction = (x0 - prox_x)/torch.norm(x0 - prox_x)

      return direction, rescale_factor
    
    def directional_grad_descent(self,x0, xk,tk,rescale_factor, direction):
        '''
            Compute the gradient of the Moreau envelope in a 1D direction
        '''

        # Compute the line parameter Tau Corresponding to the Proximal Operator
        grad_uk = self.compute_prox_tau(x0, xk, tk,rescale_factor, direction)   

        # Compute the Next Iteration Point
        #alpha = self.alpha
        #alpha = 0.01
        xk_plus1 = xk + self.alpha * grad_uk

        return xk_plus1, grad_uk

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
    
    def stopping_criteria(self,k,history):
      '''
        Stopping Criteria for HJ-MAD
      '''
      xk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist, tk_hist = history

      if xk_error_hist[k] < self.tol:
          if self.verbose:
            print('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
            print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          return True
      elif k==self.max_iters:
        if self.verbose:
          print('HJ-MAD failed to converge with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
          print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          print('Used fixed time = ', self.fixed_time)
          return True
      
      if k > 0 and np.abs(torch.norm(xk_hist[k] - xk_hist[k-1])) < self.saturate_tol*torch.norm(xk_hist[k-1]): 
        if self.verbose:
          print('HJ-MAD converged due to error saturation with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
          print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
        return True
    #   elif k > 10 and np.sum(np.diff(xk_error_hist[k-10:k+1]) > 0) > 3: # TODO: Needs to be Removed and Replaced with stopping criterion below
    #     if self.verbose:
    #       print('HJ-MAD stopped due to non-monotonic error decrease with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
    #       print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
    #     return True
    #   elif k > 20 and torch.std(fk_hist[k-20:k+1]) < self.tol:
    #     tk_hist = tk_hist[0:k+1]
    #     xk_hist = xk_hist[0:k+1,:]
    #     xk_error_hist = xk_error_hist[0:k+1]
    #     rel_grad_uk_norm_hist = rel_grad_uk_norm_hist[0:k+1]
    #     fk_hist               = fk_hist[0:k+1]
    #     print('HJ-MAD converged due to oscillating fk with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
    #     print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*int_samples)
    #     return True
    #   elif k>1  and rel_grad_uk_norm_hist[k] < 1:
    #     if self.verbose:
    #         print('HJ-MAD stopped due to small relative gradient norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
    #         print('iter = ', k, ', number of function evaluations = ', len(fk_hist)*self.int_samples)
    #     return True

    def run_in_best_direction(self, x0, t_init):

      int_samples           = self.int_samples

      # Initialize History
      xk_hist = torch.zeros(self.max_iters+1, self.n_features)
      xk_error_hist = torch.zeros(self.max_iters+1)
      rel_grad_uk_norm_hist = torch.zeros(self.max_iters+1)
      fk_hist = torch.zeros(self.max_iters+1)
      tk_hist = torch.zeros(self.max_iters+1)

      # Find the best direction to start
      direction, rescale_factor  = self.find_direction(x0,t_init)


      # Initialize Position Variables
      xk    = x0
      x_opt = xk
      tk = t_init

      # Initialize tk
      rel_grad_uk_norm      = 1.0

      # Initialize Acceleration Variables
      if self.momentum is not None:
        xk_minus_1 = xk
      
      # Initialize Gradient Variables
      _, grad_uk = self.directional_grad_descent(x0,xk,tk,rescale_factor,direction)
      grad_uk_norm      = torch.norm(grad_uk)

      fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} '
      fmt += ' | |grad_uk| = {:6.2e} | tk = {:6.2e}'
      if self.verbose:
        print('-------------------------- RUNNING HJ-MAD ---------------------------')
        print('dimension = ', self.n_features, 'n_samples = ', int_samples)

      k = 0
      while True:
        # Store History
        xk_hist[k, :] = xk
        rel_grad_uk_norm_hist[k] = rel_grad_uk_norm
        xk_error_hist[k] = torch.norm(xk - self.x_true)
        tk_hist[k] = tk
        fk_hist[k] = self.f(xk.view(1, self.n_features))

        if self.verbose:
          print(fmt.format(k+1, fk_hist[k], xk_error_hist[k], rel_grad_uk_norm_hist[k], tk))

        # # Update Optimal x Value
        if self.f(xk.view(1, self.n_features)) < self.f(x_opt.view(1, self.n_features)):
            x_opt = xk # Only update if the function value is smaller than the previous one

        # Check for Convergence
        if self.stopping_criteria(k, [xk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist, tk_hist]):
          break
        
        grad_uk_norm_old  = torch.norm(grad_uk)

        # Accelerate Gradient Descent if momentum is not none
        if self.momentum is not None and k > 0:  # when k=0 go to else with as x_{-1} = x_0
            yk = xk + self.momentum * (xk - xk_minus_1) # Check if this is on the right line
            xk_minus_1 = xk.clone()
        else:
            yk = xk.clone()

        # Perform Gradient Descent
        xk, grad_uk = self.directional_grad_descent(x0,yk, tk,rescale_factor, direction)

        if np.isnan(self.f(xk.view(1, self.n_features))):
          print('NAN Detected')
          break

        # Compute Relative Gradients
        grad_uk_norm      = torch.norm(grad_uk)
        rel_grad_uk_norm  = grad_uk_norm / (grad_uk_norm_old + 1e-12) # TODO: Check if this causes issues!

        k += 1
      # x_opt = xk
      if self.plot:
        self.plot_1d_descent(x0.numpy(), x_opt.numpy(),tk, direction.numpy())
      return x_opt, [xk_hist[:k,:], xk_error_hist[:k], rel_grad_uk_norm_hist[:k], fk_hist[:k], tk_hist[:k]]
    
    def run(self, x0, num_cycles=2):
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
        xk = x0.clone()
        tk    = self.t_vec[0]

        full_xk_hist = []
        full_fk_hist = []
        full_xk_error_hist = []

        for cycle in range(num_cycles):
            if self.verbose:
                print(f"Cycle {cycle + 1}/{num_cycles}")

            # Optimize In The Initial "Best Direction"
            x_opt, history = self.run_in_best_direction(xk,tk)


            if self.f(x_opt.view(1,-1)) < self.f(xk.view(1,-1)):
                xk = x_opt

            # Store History
            xk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist, _ = history
            full_xk_hist.extend(xk_hist.numpy())
            full_xk_error_hist.extend(xk_error_hist.numpy())
            full_fk_hist.extend(fk_hist.numpy())

            if torch.norm(xk - self.x_true) < self.tol:
                print(f'HJ-MAD-CD converged. Error: {torch.norm(xk - self.x_true):.3f}, tolerance: {self.tol}.')
                X_opt = xk
                return X_opt, full_xk_hist, full_xk_error_hist, full_fk_hist
            
            if self.fixed_time == False:
                tk = self.update_time(tk, rel_grad_uk_norm_hist[-1])
                

        X_opt = xk
        return X_opt, full_xk_hist, full_xk_error_hist, full_fk_hist
    
    def plot_1d_descent(self, x0, xk, tk, direction, domain=(-15, 15), num_points=1000):
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

        tau_vals = np.linspace(-40, 20, num_points)
        # Create x_vals with shape (num_points, 2)
        x_vals = np.zeros((num_points, 2))

        # Compute x_vals by adding tau * direction for each component of x0
        x_vals[:, 0] = x0[0] + tau_vals * direction[0]
        x_vals[:, 1] = x0[1] + tau_vals * direction[1]

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

        tau_vals = np.linspace(-40, 20, num_points)
        # Create x_vals with shape (num_points, 2)
        x_vals = np.zeros((num_points, 2))

        # Compute x_vals by adding tau * direction for each component of x0
        x_vals[:, 0] = x0[0] + tau_vals * direction[0]
        x_vals[:, 1] = x0[1] + tau_vals * direction[1]

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
    