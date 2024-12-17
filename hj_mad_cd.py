
import matplotlib.pyplot as plt
import numpy as np
import torch

epsilon_double = np.finfo(np.float64).eps

seed   = 30
torch.manual_seed(seed)

import concurrent.futures

from scipy.special import roots_hermite

# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------

class HJ_MAD:
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
    def __init__(self, f, x_true, delta=0.1, int_samples=100, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, theta=0.9, beta=[0.9], eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, 
                 verbose=True,rescale0=1e-1, momentum=None,saturate_tol=1e-9, integration_method='MC',
                 rootsGHQ=None):
      
      self.delta            = delta
      self.f                = f
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
      if rootsGHQ is not None:
        self.rootsGHQ = rootsGHQ

      
      # check that alpha is in right interval
      assert(alpha >= 1-np.sqrt(eta_vec[0]))
      assert(alpha <= 1+np.sqrt(eta_vec[1]))

    def compute_grad_uk_MC(self,x, t, k, dim=slice(None)):
      """
        Compute the gradient of Moreau Envelope of f using Monte Carlo Integration.
        
        Parameters:
            x (float): Point at which to compute the gradient.
            t (float): Time parameter.
            dim (int): Dimension to compute the gradient.
        
        Returns:
            grad (float): Computed gradient.
        """
      
      underflow = 1e-15
      overflow = 1e15
      rescale_factor = self.rescale0
      min_rescale=1e-10
      max_rescale=1e2
      iterations = 0

      while True:
        n_features = x.shape[0]
        standard_dev = np.sqrt(self.delta*t/rescale_factor)

        #samples = self.int_samples
        
        #y = standard_dev * torch.randn(samples, n_features) + x
        y = x.expand(self.int_samples, n_features).clone()

        if dim == slice(None):  # randomize all n_features of y
          y = standard_dev * self.z + x#torch.randn(self.int_samples, n_features) + x
        else:
          y = x.expand(self.int_samples, n_features).clone()
          # Replace the specified column with random samples
          y[:, dim] = standard_dev * self.z + x[dim] # torch.randn(self.int_samples) + x[dim]

        f_values = self.f(y)

        rescaled_exponent = -rescale_factor*f_values/self.delta
        # Remove the maximum exponent to prevent overflow
        max_exponent = torch.max(rescaled_exponent)  # Find the maximum exponent
      
        # max_abs_idx = torch.argmax(torch.abs(rescaled_exponent))  # Get the index of the maximum absolute value
        # max_exponent = rescaled_exponent[max_abs_idx]  # Retrieve the value from the original tensor

        shifted_exponent = rescaled_exponent - max_exponent

        exp_term = torch.exp(shifted_exponent)
        v_delta       = torch.mean(exp_term)

        # print(f'v_delta = {v_delta}')

        if (v_delta >= underflow and v_delta <= overflow) or rescale_factor < min_rescale:
          break

        # Adjust rescale factor and increment iteration count
        rescale_factor /= 2
        iterations += 1

      print(f"  {iterations=}")
      print(f"  {rescale_factor=}")
      print(f"  Variance = {self.delta*t/rescale_factor}")
      print(f"  v_delta = mean(exp(-rescale_factor*f(y)*self.delta - max) = {v_delta}")
      print(f"  max_exponent = {max_exponent}")

      # Increase intial rescale factor if it is too small to better utilize samples
      if iterations == 0:
        self.rescale0 = min(2*rescale_factor, max_rescale)
      else:
        self.rescale0 = rescale_factor

      # THIS IS IMPORTANT FOR TUNING INITIAL RESCALE FACTOR
      # print(f'Loops to find rescale factor: {loop_iterations}')
      numerator = y*exp_term.view(self.int_samples, 1)
      numerator = torch.mean(numerator, dim=0)
        
      # Compute Grad U
      # TODO: Check if this is correct
      grad_uk = (x -  numerator/(v_delta)) # the t gets canceled with the update formula

      grad_norm = torch.norm(grad_uk/(t/rescale_factor)).item()

      if grad_norm < 1e-5 and k >500:
        self.theta *= 0.9
      print(f"{k=}, {self.theta=}, {grad_norm=}")

      # Compute Moreau envelope
      #uk = -self.delta * torch.log(v_delta)

      # Compute Estimated prox_xk
      prox_xk = numerator / (v_delta)

      return grad_uk, prox_xk
  

    def compute_grad_uk_GHQ(self, x, t, dim):
        """
        Compute the gradient of Moreau Envelope of f using Gauss-Hermite quadrature.
        
        Parameters:
            x (float): Point at which to compute the gradient.
            t (float): Time parameter.
            dim (int): Dimension to compute the gradient.
        
        Returns:
            grad (float): Computed gradient.
        """

        n_features = x.shape[0]
        min_rescale=1e-15
        max_rescale=1e2
        underflow = 1e-15
        overflow = 1e15

        z, weights = self.rootsGHQ

        # Define rescale_factor for preventing overflow/underflow
        rescale_factor = self.rescale0
        rescale_counter = 0

        # The device on which the computation is done
        device = x.device

        # Repeat x for all samples
        y = x.clone().expand(self.int_samples,n_features)
        y = y.clone().contiguous()

        while True:
          # Rescale t
          t_rescaled = t/rescale_factor

          # Compute the roots and weights for the Hermite quadrature

          # Compute the integral of the exponential for f in 1 Dimension
          y_1D = x[dim] - z*np.sqrt(2*self.delta*t_rescaled) # Size (int_samples,1)
          y[:,dim] = y_1D # Update y only in the dimension we are moving.

          rescaled_f_exponent = - rescale_factor*self.f(y)/ self.delta
          max_exponent = torch.max(rescaled_f_exponent)  # Find the maximum exponent
          shifted_exponent = rescaled_f_exponent - max_exponent
          F_exp = torch.exp(shifted_exponent)


          # Compute the Denominator Integral
          v_delta = torch.sum(weights * F_exp) # * (1 / np.sqrt(np.pi))  # TODO: fix the bug if this is negative

          # Make sure over/underflow does not occur in the Denominator
          if (v_delta >= underflow and v_delta <= overflow) or rescale_factor < min_rescale: 
            break
          
          # Adjust rescale factor and increment iteration count
          rescale_factor /= 2
          rescale_counter += 1
        
        # Increase intial rescale factor if it is too small to better utilize samples
        if rescale_counter == 0:
          rescale_factor = min(2*rescale_factor, max_rescale)

        self.rescale0 = rescale_factor

        print(f"{rescale_counter=}")
        print(f"{self.rescale0=}")

        # Compute Numerator Integral
        grad_v_delta_F = z * F_exp
          
        # numerator = - np.sqrt(2/(self.delta*t_rescaled)) * torch.sum(weights * grad_v_delta_F) # * (1 / np.sqrt(np.pi)) 
        numerator = np.sqrt(2/(self.delta*t_rescaled)) * torch.sum(weights * grad_v_delta_F) # * (1 / np.sqrt(np.pi)) 

        # Compute Gradient in 1D
        grad_uk_1D = self.delta * numerator / v_delta
        grad_uk = torch.zeros(n_features, dtype=torch.float64, device=device)
        grad_uk[dim] = grad_uk_1D

        # Compute Prox_xk
        prox_xk = x - t_rescaled*grad_uk
        
        return t_rescaled*grad_uk, prox_xk


    def gradient_descent(self, xk, tk,k, update_dim=slice(None)):
        # Compute prox and gradient
        if self.integration_method == 'MC':
          grad_uk, prox_xk = self.compute_grad_uk_MC(xk, tk,k, update_dim)

          # Perform gradient descent update
          xk_plus1 = xk.clone()
          xk_plus1[update_dim] = xk[update_dim] - self.alpha * (xk[update_dim] - prox_xk[update_dim])

        elif self.integration_method == 'GHQ':
          grad_uk, prox_xk =self.compute_grad_uk_GHQ(xk, tk, update_dim)
        
          # Perform gradient descent update
          xk_plus1 = xk - self.alpha * grad_uk

        return xk_plus1, grad_uk, prox_xk

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
    
    def stopping_criteria(self,k,cd,history):
      '''
        Stopping Criteria for HJ-MAD and HJ-MAD-CD
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
      if cd:
        if k > 0 and np.abs(torch.norm(xk_hist[k] - xk_hist[k-1])) < self.saturate_tol*torch.norm(xk_hist[k-1]): 
          if self.verbose:
            print('HJ-MAD converged due to error saturation with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
            print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          return True
        # elif k > 10 and np.sum(np.diff(xk_error_hist[k-10:k+1]) > 0) > 3: # TODO: Needs to be Removed and Replaced with stopping criterion below
        #   if self.verbose:
        #     print('HJ-MAD stopped due to non-monotonic error decrease with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
        #     print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
        #   return True
        elif k > 20 and torch.std(fk_hist[k-20:k+1]) < self.tol:
          print('HJ-MAD converged due to oscillating fk with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
          print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist)*self.int_samples)
          return True

    
    def run(self, x0, cd=False, update_dim=slice(None)):
      """
      Run the HJ-MAD algorithm to minimize the function.

      Parameters:
      x0 (torch.Tensor): Initial guess for the minimizer.
      cd (bool): Coordinate descent flag.
      update_dim (slice): Dimension to update.

      Returns:
      x_opt (torch.Tensor): Optimal x value approximation.
      xk_hist (torch.Tensor): Update history.
      tk_hist (torch.Tensor): Time history.
      xk_error_hist (torch.Tensor): Error to true solution history.
      rel_grad_uk_norm_hist (torch.Tensor): Relative grad norm history of Moreau envelope.
      fk_hist (torch.Tensor): Function value history.
      """
      # Dimensions of x0
      n_features = x0.shape[0]

      # Initialize history tensors
      xk_hist = torch.zeros(self.max_iters, n_features)
      xk_error_hist = torch.zeros(self.max_iters)
      rel_grad_uk_norm_hist = torch.zeros(self.max_iters)
      fk_hist = torch.zeros(self.max_iters)
      tk_hist = torch.zeros(self.max_iters)

      # Initialize iteration variables x and t
      xk = x0
      x_opt = xk
      tk = self.t_vec[0]

      # Set up Momentum
      if self.momentum is not None:
        xk_minus_1 = xk

      rel_grad_uk_norm = 1.0

      if self.integration_method == 'NMC' or self.integration_method == 'MC':
        if update_dim == slice(None):  # randomize all n_features of y
          self.z = torch.randn(self.int_samples,n_features)
        else:
          self.z = torch.randn(self.int_samples)

      fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} | |grad_uk| = {:6.2e} | tk = {:6.2e}'
      if self.verbose:
        print('-------------------------- RUNNING HJ-MAD ---------------------------')
        print('dimension = ', n_features, 'n_samples = ', self.int_samples)

      # Compute initial gradient
      k =0
      _ , grad_uk, _ = self.gradient_descent(xk, tk,k, update_dim)


      for k in range(self.max_iters):
        # Store current state in history
        xk_hist[k, :] = xk
        rel_grad_uk_norm_hist[k] = rel_grad_uk_norm
        xk_error_hist[k] = torch.norm(xk - self.x_true)
        tk_hist[k] = tk
        fk_hist[k] = self.f(xk.view(1, n_features))

        if self.verbose:
          print(fmt.format(k + 1, fk_hist[k], xk_error_hist[k], rel_grad_uk_norm_hist[k], tk))

        # Check for convergence
        if self.stopping_criteria(k, cd, [xk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist, tk_hist]):
          break

        if k > 0 and fk_hist[k] < fk_hist[k - 1]:
          x_opt = xk

        grad_uk_norm_old = torch.norm(grad_uk)

        # Accelerate gradient descent if momentum is not None
        if self.momentum is not None and k > 0:
          yk = xk.clone()
          yk[update_dim] = xk[update_dim] + self.momentum * (xk[update_dim] - xk_minus_1[update_dim])
          xk_minus_1[update_dim] = xk[update_dim]
        else:
          yk = xk.clone()

        # Perform gradient descent
        xk, grad_uk, prox_xk = self.gradient_descent(yk, tk,k, update_dim)

        # Compute relative gradients
        grad_uk_norm = torch.norm(grad_uk)
        rel_grad_uk_norm = grad_uk_norm / (grad_uk_norm_old + 1e-12)

        # Update tk
        if not self.fixed_time:
          tk = self.update_time(tk, rel_grad_uk_norm)

  
      return x_opt, xk_hist[0:k+1,:], tk_hist[0:k+1], xk_error_hist[0:k+1], rel_grad_uk_norm_hist[0:k+1], fk_hist[0:k+1]


# ------------------------------------------------------------------------------------------------------------
# Coordinate HJ MAD
# ------------------------------------------------------------------------------------------------------------

class HJ_MAD_CoordinateDescent(HJ_MAD):
    """
    Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) Coordinate Descent for 2D functions.

    This class extends the HJ_MAD algorithm to perform coordinate descent for 2D functions. It alternates between 
    optimizing each coordinate while keeping the other fixed, treating the function as a 1D function for each run 
    and using the previous solution in the next run.

    Attributes:
        f (callable): The function to be minimized.
        x_true (torch.Tensor): The true global minimizer.
        delta (float): Coefficient of the viscous term in the HJ equation.
        int_samples (int): Number of samples used to approximate expectation in the heat equation solution.
        t_vec (list): Time vector containing [initial time, minimum time allowed, maximum time].
        max_iters (int): Maximum number of iterations.
        tol (float): Stopping tolerance.
        alpha (float): Step size.
        beta (float): Exponential averaging term for gradient beta.
        eta_vec (list): Vector containing [eta_minus, eta_plus].
        theta (float): Parameter used to update tk.
        fixed_time (bool): Whether to use adaptive time.
        verbose (bool): Whether to print progress.
        rescale0 (float): Initial rescale factor.
        momentum (float): Momentum term for acceleration.

    Methods:
        run(x0, num_cycles): Runs the coordinate descent optimization process.
    """

    def __init__(self, f, x_true, delta=0.1, int_samples=100, t_vec=[1.0, 1e-3, 1e1], max_iters=5e4,
                 tol=5e-2, theta=0.9, beta=[0.9], eta_vec=[0.9, 1.1], alpha=1.0, fixed_time=False,
                 plot=False, verbose=True, rescale0=1e-1, momentum=None,saturate_tol=1e-9,integration_method='MC'):
        self.tol = tol
        self.plot = plot

        if integration_method == 'GHQ':
            device = x_true.device
            z, weights = roots_hermite(int_samples)
            z = torch.tensor(z, dtype=torch.float64, device=device)
            weights = torch.tensor(weights, dtype=torch.float64, device=device)
            rootsGHQ = (z, weights)
        else:
            rootsGHQ = None

        super().__init__(f=f, x_true=x_true, delta=delta, int_samples=int_samples, t_vec=t_vec, max_iters=max_iters,
                         tol=self.tol, alpha=alpha, beta=beta, eta_vec=eta_vec, theta=theta, fixed_time=fixed_time,
                         verbose=verbose, rescale0=rescale0, momentum=momentum,saturate_tol=saturate_tol,integration_method=integration_method,
                         rootsGHQ=rootsGHQ)

    def plot_1d_descent(self, xk, xk_new, dim, domain=(-15, 15), num_points=1000):
        """
        Plots the 1D descent for the current dimension.

        Args:
            xk (torch.Tensor): Current position.
            dim (int): The current dimension being optimized.
            domain (tuple): The range over which to vary the current dimension.
            num_points (int): Number of points to sample in the domain.
        """
        x_vals = np.linspace(domain[0], domain[1], num_points)
        f_vals = []
        h_vals = []

        for x in x_vals:
            xk_varied = xk.clone()
            xk_varied[dim] = x
            f_vals.append(self.f(xk_varied.unsqueeze(0)).item())
            h_vals.append(self.f(xk_varied.unsqueeze(0)).item() + 1/(2*self.t_vec[0])*torch.norm(xk_varied-xk)**2)

        std_dev = np.sqrt(self.delta * self.t_vec[0]/self.rescale0)
        std_dev_minus = xk.clone()
        std_dev_plus = xk.clone()
        std_dev_minus[dim] -= std_dev
        std_dev_plus[dim] += std_dev
        # print(f"{self.delta} * {self.t_vec[0]} = {self.delta * self.t_vec[0]}")
        # print(f'Std Dev: {std_dev}, Std Dev Minus: {std_dev_minus}, Std Dev Plus: {std_dev_plus}')

        plt.figure()
        plt.plot(x_vals, f_vals, '-', color='black', label=f'f(x) Dimension {dim + 1}')
        plt.plot(x_vals, h_vals, '-', color='blue', label=r'$f(x) + \frac{1}{2t_0} ||x - x_k||^2$')
        plt.plot(xk[dim], self.f(xk.unsqueeze(0)).item(), '*', color='red', label=f'xk Dimension {dim + 1}')
        plt.plot(xk_new[dim], self.f(xk_new.unsqueeze(0)).item(), '*', color='green', label=f'New xk Dimension {dim + 1}')
        plt.plot(std_dev_minus[dim].item(), self.f(std_dev_minus.unsqueeze(0)).item(), 'x', color='purple', label='Std Devs')
        plt.plot(std_dev_plus[dim].item(), self.f(std_dev_plus.unsqueeze(0)).item(), 'x', color='purple')
        plt.xlabel(f'Dimension {dim + 1}')
        plt.ylabel('Function Value')
        plt.title(f'1D Descent for Dimension {dim + 1}')
        plt.legend()
        plt.show()

    def run(self, x0, num_cycles=10):
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
        n_features = x0.shape[0]
        CD_xk_hist = torch.zeros(n_features*num_cycles+1, n_features)
        CD_xk_hist[0,:]    = xk
        full_xk_hist = []
        full_fk_hist = []
        full_xk_error_hist = []

        # x_opt, xk_hist, _, xk_error_hist, _, _ = super().run(xk,cd=True)
        # xk = x_opt.clone()
        # full_history.extend(xk_hist)
        # xk_error_hist_MAD.extend(xk_error_hist)

        for cycle in range(num_cycles):
            dims = list(range(n_features))
            #dims = [1, 0]

            # # Randomly select 50% of the dimensions
            # dims = random.sample(range(n_features), k=n_features // 2)
            dim_count=0
            for dim in dims:
                # if dim == 1:
                #     self.t_vec[0] = 800
                #     self.t_vec[1] = 0.1
                #     self.t_vec[2] = 1000
                #     self.delta = 0.1
                #     self.rescale0 = 1

                # Plot the 1D descent for the current dimension
                xk_prev = xk.clone()
                
                if self.verbose:
                    print(f"Cycle {cycle + 1}/{num_cycles} and Dimension {dim + 1}/{n_features}")

                # Optimize with respect to the first coordinate
                xk, xk_hist, tk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist = super().run(xk,cd=True, update_dim=dim)

                if self.plot:
                    self.plot_1d_descent(xk_prev,xk, dim)
     
                full_xk_hist.extend(xk_hist.numpy())
                full_xk_error_hist.extend(xk_error_hist.numpy())
                full_fk_hist.extend(fk_hist.numpy())

                if  xk_error_hist[-1] < self.tol:
                    print(f'HJ-MAD-CD converged. Error: {xk_error_hist[-1]:.3f}, tolerance: {self.tol}.')
                    CD_xk_hist[cycle+1,:]    = xk
                    X_opt = xk
                    return X_opt, CD_xk_hist,full_xk_hist, full_xk_error_hist, full_fk_hist
                
                CD_xk_hist[cycle+dim_count+1,:]    = xk
                dim_count+=1

            # if cycle > 0 and cycle % 3 == 0:
            #     self.int_samples *= 2

            # CD_xk_hist[cycle+1,:]    = xk

        X_opt = xk
        return X_opt, CD_xk_hist,full_xk_hist, full_xk_error_hist, full_fk_hist
    

# ------------------------------------------------------------------------------------------------------------
# Coordinate HJ MAD Parallel
# ------------------------------------------------------------------------------------------------------------

class HJ_MAD_CoordinateDescent_parallel(HJ_MAD):
    """
    Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) Coordinate Descent for 2D functions.

    This class extends the HJ_MAD algorithm to perform coordinate descent for 2D functions. It alternates between 
    optimizing each coordinate while keeping the other fixed, treating the function as a 1D function for each run 
    and using the previous solution in the next run.

    Attributes:
        f (callable): The function to be minimized.
        x_true (torch.Tensor): The true global minimizer.
        delta (float): Coefficient of the viscous term in the HJ equation.
        int_samples (int): Number of samples used to approximate expectation in the heat equation solution.
        t_vec (list): Time vector containing [initial time, minimum time allowed, maximum time].
        max_iters (int): Maximum number of iterations.
        tol (float): Stopping tolerance.
        alpha (float): Step size.
        beta (float): Exponential averaging term for gradient beta.
        eta_vec (list): Vector containing [eta_minus, eta_plus].
        theta (float): Parameter used to update tk.
        fixed_time (bool): Whether to use adaptive time.
        verbose (bool): Whether to print progress.
        rescale0 (float): Initial rescale factor.
        momentum (float): Momentum term for acceleration.

    Methods:
        run(x0, num_cycles): Runs the coordinate descent optimization process.
    """

    def __init__(self, f, x_true, delta=0.1, int_samples=100, t_vec=[1.0, 1e-3, 1e1], max_iters=5e4,
                 tol=5e-2, theta=0.9, beta=[0.9], eta_vec=[0.9, 1.1], alpha=1.0, fixed_time=False,
                 verbose=True,plot=False, rescale0=1e-1, momentum=None, saturate_tol=1e-9):
        self.tol = tol
        self.plot = plot
        super().__init__(f=f, x_true=x_true, delta=delta, int_samples=int_samples, t_vec=t_vec, max_iters=max_iters,
                         tol=self.tol, alpha=alpha, beta=beta, eta_vec=eta_vec, theta=theta, fixed_time=fixed_time,
                         verbose=verbose, rescale0=rescale0, momentum=momentum, saturate_tol=saturate_tol)

    def plot_1d_descent(self, xk, dim, domain=(-10, 10), num_points=100):
        """
        Plots the 1D descent for the current dimension.

        Args:
            xk (torch.Tensor): Current position.
            dim (int): The current dimension being optimized.
            domain (tuple): The range over which to vary the current dimension.
            num_points (int): Number of points to sample in the domain.
        """
        x_vals = np.linspace(domain[0], domain[1], num_points)
        f_vals = []

        for x in x_vals:
            xk_varied = xk.clone()
            xk_varied[dim] = x
            f_vals.append(self.f(xk_varied.unsqueeze(0)).item())

        std_dev = np.sqrt(self.delta * self.t_vec[0])
        std_dev_minus = xk.clone()
        std_dev_plus = xk.clone()
        std_dev_minus[dim] -= std_dev
        std_dev_plus[dim] += std_dev

        plt.figure()
        plt.plot(x_vals, f_vals, '-', color='black', label=f'f(x) Dimension {dim + 1}')
        plt.plot(xk[dim].item(), self.f(xk.unsqueeze(0)).item(), '*', color='red', label=f'xk Dimension {dim + 1}')
        plt.plot(std_dev_minus[dim].item(), self.f(std_dev_minus.unsqueeze(0)).item(), 'x', color='purple', label='Std Dev Minus')
        plt.plot(std_dev_plus[dim].item(), self.f(std_dev_plus.unsqueeze(0)).item(), '*', color='purple', label='Std Dev Plus')
        plt.xlabel(f'Dimension {dim + 1}')
        plt.ylabel('Function Value')
        plt.title(f'1D Descent for Dimension {dim + 1}')
        plt.legend()
        plt.show()

    def optimize_dimension(self, xk, dim):
        """
        Optimize with respect to a single dimension.

        Args:
            xk (torch.Tensor): Current position.
            dim (int): The dimension to optimize.

        Returns:
            torch.Tensor: Updated position for the dimension.
            list: History of positions for the dimension.
            list: History of errors for the dimension.
            list: History of function values for the dimension.
        """
        xk, xk_hist, tk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist = super().run(xk, cd=True, update_dim=dim)
        return xk, xk_hist, xk_error_hist, fk_hist

    def run(self, x0, num_cycles=10) -> Tuple[torch.Tensor, torch.Tensor, List, List, List]:
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
        n_features = x0.shape[0]
        CD_xk_hist = torch.zeros(n_features * num_cycles + 1, n_features)
        CD_xk_hist[0, :] = xk
        full_xk_hist = []
        full_fk_hist = []
        full_xk_error_hist = []

        for cycle in range(num_cycles):
            dims = list(range(n_features))
            dim_count = 0
            print("Cycle", cycle + 1)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.optimize_dimension, xk.clone(), dim): dim for dim in dims}
                results = {dim: future.result() for future, dim in futures.items()}

            for dim, (xk_dim, xk_hist, xk_error_hist, fk_hist) in results.items():
                if self.verbose:
                    print(f"Cycle {cycle + 1}/{num_cycles} and Dimension {dim + 1}/{n_features}")

                if self.plot:
                    self.plot_1d_descent(xk, dim)

                xk[dim] = xk_dim[dim]
                full_xk_hist.extend(xk_hist.numpy())
                full_xk_error_hist.extend(xk_error_hist.numpy())
                full_fk_hist.extend(fk_hist.numpy())

                if xk_error_hist[-1] < self.tol:
                    if self.verbose:
                        print(f'HJ-MAD-CD converged. Error: {xk_error_hist[-1]:.3f}, tolerance: {self.tol}.')
                    CD_xk_hist[cycle + 1, :] = xk
                    X_opt = xk
                    return X_opt, CD_xk_hist, full_xk_hist, full_xk_error_hist, full_fk_hist

                CD_xk_hist[cycle + dim_count + 1, :] = xk
                dim_count += 1
            print(f"Error = {torch.norm(xk - self.x_true)}")
            
            if xk_error_hist[-1] < self.tol:
                if self.verbose:
                    print(f'HJ-MAD-CD converged. Error: {xk_error_hist[-1]:.3f}, tolerance: {self.tol}.')
                CD_xk_hist[cycle + 1, :] = xk
                X_opt = xk
                return X_opt, CD_xk_hist, full_xk_hist, full_xk_error_hist, full_fk_hist
        X_opt = xk
        return X_opt, CD_xk_hist, full_xk_hist, full_xk_error_hist, full_fk_hist