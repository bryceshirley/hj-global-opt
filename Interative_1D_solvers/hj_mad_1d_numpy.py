import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, clear_output, display

# HJ Moreau Adaptive Descent

class HJ_MAD_1D_NUMPY:
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


    def __init__(self, f, x_true, x0, delta=0.1, int_samples=100, t_vec = [1.0, 1e-3, 1e1], max_iters=5e4, 
                 tol=5e-2, theta=0.9, beta=0.0, eta_vec = [0.9, 1.1], alpha=1.0, fixed_time=False, 
                 verbose=True, accelerated=False,momentum=0.5, plot_parameters = [-30,30,None],
                 sample_bool=False, trap_integration=False):
      
        self.delta            = delta
        self.f                = f
        self.int_samples      = int_samples
        self.max_iters        = max_iters
        self.tol              = tol
        self.t_vec            = t_vec
        self.theta            = theta
        self.x_true           = x_true
        self.x0               = x0
        self.beta             = beta 
        self.alpha            = alpha 
        self.eta_vec          = eta_vec
        self.fixed_time       = fixed_time
        self.verbose          = verbose
        self.accelerated      = accelerated
        self.trap_integration = trap_integration
        self.momentum         = momentum
        # Plotting Parameters
        self.plot_parameters  = plot_parameters
        self.sample_bool      = sample_bool
      
        # check that alpha is in right interval
        assert(alpha >= 1-np.sqrt(eta_vec[0]))
        assert(alpha <= 1+np.sqrt(eta_vec[1]))

    def find_rescale_factor(self, x, t, initial_rescale_factor=1, epsilon=1e-15, min_rescale=1e-10):
        """
        Finds the optimal rescale factor to ensure that the exponential term remains above epsilon.
        
        Parameters:
        - x : float or np.array
            The current point or array of points.
        - t : float
            Time or other parameter affecting the standard deviation.
        - initial_rescale_factor : float, optional
            Starting value for the rescale factor.
        - epsilon : float, optional
            Threshold for the maximum exponential term to be considered valid.
        - min_rescale : float, optional
            Minimum value for the rescale factor to prevent infinite loops.
        
        Returns:
        - tuple (float, int)
            The optimal rescale factor and the number of iterations taken to reach it.
        """

        rescale_factor = initial_rescale_factor
        iterations = 0

        while True:
            # Calculate standard deviation and generate random samples
            standard_dev = np.sqrt(self.delta * t / rescale_factor)
            y = standard_dev * np.random.randn(self.int_samples) + x

            # Compute function values and the max exponential term
            f_values = self.f(y)
            # max_f_values = np.max(f_values)
            v_delta = np.mean(np.exp(-rescale_factor * f_values / self.delta))


            # min_exponent = -rescale_factor * max_f_values / self.delta
            # v_delta_exponent = np.log(v_delta)

            # print(f"{iterations=}")
            # print(f"{min_exponent=}")
            # print(f"{v_delta_exponent=}")
            # print(f"{np.log(epsilon)=}\n")

            # Check if the maximum exponential term is within the desired range

            if v_delta >= epsilon or rescale_factor < min_rescale:
                break

            # Adjust rescale factor and increment iteration count
            rescale_factor /= 2
            iterations += 1

        return rescale_factor, iterations
    
    def compute_grad_uk(self, x: float, t: float, grad_uk_old: Optional[float] = None) -> Tuple[float, float, float, np.ndarray]:
        ''' 
        Compute the gradient of the Moreau envelope and related statistics.

        Args:
            x (float): Input point for evaluating the gradient.
            t (float): Scaling factor for smoothing (typically 'tk').
            eps (float, optional): Small constant for numerical stability. Default is 1e-14.
            old_grad_uk (Optional[float], optional): Previous gradient for momentum updates. Default is None.

        Returns:
            Tuple[float, float, float, np.ndarray]: 
                - grad_uk: Gradient of the Moreau envelope.
                - uk: Moreau envelope estimate.
                - se_uk: Standard error of 'uk'.
                - y: Random samples drawn from N(x, delta * t).
        '''
        delta = self.delta
        f = self.f

        rescale, _ = self.find_rescale_factor(x, t)
        # print(f"{rescale=}")
        # print(f"{iterations=}")

        # Compute the function of the random variable y sampled from N(x,delta*t)
        standard_dev = np.sqrt(delta * t/rescale) 
        y = standard_dev * np.random.randn(self.int_samples) + x  
        exp_term = np.exp(-rescale*f(y) / delta)

        # Compute Denominator and average over the samples (add eps for 0 error)
        v_delta = np.mean(exp_term)

         # Compute Numerator and average over the samples
        numerator = np.mean(y * exp_term)

        # Compute Estimated prox_xk
        prox_xk = numerator / v_delta

        # Compute Gradient at uk (times tk)
        grad_uk = (x - prox_xk)

        # Compute estimated uk
        uk = -delta * np.log(v_delta)


        # Compute the standard error for uk
        # Filter out non-positive values from exp_term

        valid_exp_term = exp_term[exp_term > 0]
        no_valid_exp_terms = len(valid_exp_term)
        # Compute the standard error for uk only with valid values
        if len(valid_exp_term) > 0:
            log_exp_terms = -delta * np.log(valid_exp_term)
            sample_var = np.var(log_exp_terms, ddof=1)  # Weighted variance
            se_uk = np.sqrt(sample_var / no_valid_exp_terms) 
        else:
            se_uk = 0.0
        
        # ADAM's Method
        if grad_uk_old is not None:
            grad_uk = self.beta*grad_uk_old + (1-self.beta)*grad_uk

        # if self.f(prox_xk) > self.f(x):
        #     self.delta = self.delta* 0.95


        # Return Gradient at uk, uk, prox at xk and standard error in uk sample
        return grad_uk, uk, se_uk, y
    
    def compute_grad_uk_trapezium(self, x: float, t: float, eps: float = 1e-12, grad_uk_old: Optional[float] = None) -> Tuple[float, float, float, np.ndarray]:
        ''' 
        Compute the gradient of the Moreau envelope using the trapezium rule.

        Args:
            x (float): Input point for evaluating the gradient.
            t (float): Scaling factor for smoothing (typically 'tk').
            eps (float, optional): Small constant for numerical stability. Default is 1e-14.
            old_grad_uk (Optional[float], optional): Previous gradient for momentum updates. Default is None.

        Returns:
            Tuple[float, float, np.ndarray]: 
                - grad_uk: Gradient of the Moreau envelope.
                - uk: Moreau envelope estimate.
                - y: Discretized points used in trapezium rule.
        '''
        delta = self.delta
        f = self.f
        
        # Discretize the domain for trapezium rule
        a, b = x - 10 * np.sqrt(delta * t), x + 10 * np.sqrt(delta * t)  # Define an interval around x (5 standard deviations)
        N = self.int_samples # Number of intervals
        y = np.linspace(a, b, N)
        h = (b - a) / (N - 1)

        # Define the functions to integrate
        def integrand_v_delta(y):
            """Function for v_delta integrand."""
            return np.exp(-f(y) / delta) * np.exp(-(x - y)**2 / (2 * delta * t))
        
        def integrand_grad_v_delta(y):
            """Function for ∇v_delta integrand."""
            return (x - y) / t * integrand_v_delta(y)

        # Calculate v_delta using the trapezium rule
        discrete_integrand_v_delta = integrand_v_delta(y)
        v_delta = (h / 2) * (discrete_integrand_v_delta[0] + 2 * np.sum(discrete_integrand_v_delta[1:N-1]) + discrete_integrand_v_delta[-1])
        
        # Calculate ∇v_delta using the trapezium rule
        discrete_integrand_grad_v_delta = integrand_grad_v_delta(y)
        grad_v_delta = (h / 2) * (discrete_integrand_grad_v_delta[0] + 2 * np.sum(discrete_integrand_grad_v_delta[1:N-1]) + discrete_integrand_grad_v_delta[-1])
        
        # Compute ∇u_delta using the simplified formula: -(grad_v_delta / v_delta)
        grad_uk = grad_v_delta / (v_delta+ eps)

        # Compute prox_xk
        #prox_xk = x - t*grad_uk

        # Compute estimated uk
        uk = -delta * np.log(2 * np.pi * t * v_delta+ eps)
        
        # ADAM's Method
        if grad_uk_old is not None:
            print(self.beta)
            grad_uk = self.beta * grad_uk_old + (1 - self.beta) * grad_uk

        uk_info = (t*grad_uk, uk, None, None)

        # Return Gradient at uk, uk, prox at xk and standard error in uk sample
        return uk_info


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
    
    def gradient_descent(self, xk: float, tk: float,grad_uk_old: Optional[float] = None) -> Tuple[float, Tuple[float, float, float, np.ndarray]]:
        '''
        Perform a gradient descent update using the gradient of the Moreau envelope.

        Args:
            xk (float): Current point (iteration k) where the gradient is evaluated.
            tk (float): Time scaling factor (typically 'tk') for smoothing in the gradient computation.

        Returns:
            Tuple[float, Tuple[float, float, float, np.ndarray]]:
                - xk_plus1 (float): Updated point after the gradient descent step.
                - uk_info (Tuple[float, float, float, np.ndarray]): 
                    - grad_uk (float): Computed gradient of the Moreau envelope.
                    - uk (float): Estimated Moreau envelope at 'xk'.
                    - se_uk (float): Standard error of the Moreau envelope estimate.
                    - y (np.ndarray): Random samples used in the gradient estimation.
        '''
        # Compute prox and gradient
        if self.trap_integration:
            uk_info = self.compute_grad_uk_trapezium(xk, tk,grad_uk_old=grad_uk_old)
        else:
            uk_info = self.compute_grad_uk(xk, tk,grad_uk_old=grad_uk_old)

        # uk_info is a tuple such that uk_info[0] = grad_uk

        # Perform gradient descent update
        xk_plus1 = xk - self.alpha * uk_info[0]

        return xk_plus1, uk_info

    def run(self, animate,plot_bool=True):
        xk_hist = np.zeros(self.max_iters)
        xk_error_hist = np.zeros(self.max_iters)
        rel_grad_uk_norm_hist = np.zeros(self.max_iters)
        fk_hist = np.zeros(self.max_iters)
        tk_hist = np.zeros(self.max_iters)

        xk = self.x0
        x_opt = xk
        tk = self.t_vec[0]
        t_max = self.t_vec[2]

        uk_info  = self.compute_grad_uk(xk, tk)
        grad_uk = uk_info[0]
        grad_uk_old = None
        
        rel_grad_uk_norm = 1.0

        if self.accelerated:
            xk_minus_1 = xk
            momentum = self.momentum

        if self.beta != 0.0:
            grad_uk_old = grad_uk


        fmt = '[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} '
        fmt += ' | |grad_uk| = {:6.2e} | tk = {:6.2e}'
        for k in range(self.max_iters):
            
            # Update History
            xk_hist[k] = xk
            rel_grad_uk_norm_hist[k] = rel_grad_uk_norm
            xk_error_hist[k] = np.linalg.norm(xk - self.x_true)
            tk_hist[k] = tk
            fk_hist[k] = self.f(xk)


            if animate:
                self.plot(k, xk, tk,xk_error_hist[k],uk_info=uk_info)
                time.sleep(0.5)

            if self.verbose:
                print(fmt.format(k + 1, fk_hist[k], xk_error_hist[k], rel_grad_uk_norm_hist[k], tk))

            if xk_error_hist[k] < self.tol:
                tk_hist = tk_hist[:k + 1]
                xk_hist = xk_hist[:k + 1]
                xk_error_hist = xk_error_hist[:k + 1]
                rel_grad_uk_norm_hist = rel_grad_uk_norm_hist[:k + 1]
                fk_hist = fk_hist[:k + 1]

                print('-------------------------- HJ-MAD RESULTS ---------------------------')
                print('HJ-MAD converged with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
                print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist) * self.int_samples)
                
                break
            elif k == self.max_iters - 1:
                print('-------------------------- HJ-MAD RESULTS ---------------------------')
                print('HJ-MAD failed to converge with rel grad norm {:6.2e}'.format(rel_grad_uk_norm_hist[k]))
                print('iter = ', k, ', number of function evaluations = ', len(xk_error_hist) * self.int_samples)
                print('Used fixed time = ', self.fixed_time)

            if k > 0:
                if fk_hist[k] < fk_hist[k - 1]:
                    x_opt = xk

            grad_uk_norm_old  =  np.linalg.norm(grad_uk)

            # Accelerate
            if self.accelerated and k > 0:
                yk = xk + momentum * (xk - xk_minus_1)
                xk_minus_1 = xk
            else:
                yk=xk


            # Perform GD
            xk, uk_info = self.gradient_descent(yk,tk,grad_uk_old=grad_uk_old)

            grad_uk = uk_info[0]
            
            if self.beta != 0.0:
                grad_uk_old = grad_uk
    
            # Compute Relative Grad Uk
            grad_uk_norm = np.linalg.norm(grad_uk)
            rel_grad_uk_norm = grad_uk_norm / (grad_uk_norm_old + 1e-12)

            if not self.fixed_time:
                tk = self.update_time(tk, rel_grad_uk_norm)


        if plot_bool:
            self.plot(k,xk, tk,xk_error_hist[k],plot_bool)

        algorithm_hist = (xk_hist, tk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist)
                     
        return x_opt, algorithm_hist
    

    def plot(self, k: int, xk: float, tk: float, error: float, plot_bool: bool = True, 
         uk_info: Optional[Tuple[float, float, float, int]] = None) -> None:
        intervalx_a, intervalx_b, plot_output = self.plot_parameters

        # Use the selected global function to compute f_values
        x_range = np.linspace(intervalx_a, intervalx_b, 500)  # Adjust based on selected function
        f_values = np.array([self.f(x) for x in x_range])

        # Compute infor about Moreau envelope at xk
        if uk_info is None:
            uk_info = self.compute_grad_uk(xk, tk)

        grad_uk, uk, se_uk, samples = uk_info
            
        prox_xk = xk - grad_uk

        #xk_plus1 = xk - self.alpha * grad_uk
        #estimated_moreau_value = self.f(prox_xk) + (1 / (2 * tk)) * ((prox_xk - xk) ** 2)



        # Compute the Error
        #xk_plus1 = xk - self.alpha * grad_uk
        if k==0:
            error = np.linalg.norm(self.x0 - self.x_true) 

        # The function that prox minimizes
        prox_func_values = np.array([self.f(x) + (1 / (2 * tk)) * (x - xk) ** 2 for x in x_range])

        # Plot f(x) 
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, f_values, label='f(x)', color='black')

        # Plot the function that prox minimizes at xk and its estimated minima
        plt.plot(x_range, prox_func_values, label=r'$f(x) + \frac{1}{2T} (x - x_k)^2$', color='orange')
        plt.scatter(prox_xk, self.f(prox_xk) + (1 / (2 * tk)) * (prox_xk - xk) ** 2,
                    facecolors='none',edgecolors='orange', label=r'Estimated Prox at $ x_k$, $prox_{tf}(x_k)\approx x_k-t_k\nabla u^{\delta}(x_k)$ and' +
                    f'\nthe Estimated Global Minima of ' + r'$f(x) + \frac{1}{2t_k} (x - x_k)^2$',
                    s=100, zorder=4, marker='s')
        
        # Samples
        if self.sample_bool:
            samples_func_values = np.array([self.f(sample) for sample in samples])# + (1 / (2 * tk)) * (sample - xk) ** 2
            plt.scatter(samples, samples_func_values,color='blue', label=r'Samples', zorder=4, marker='*')
        

        # Plot Moreau Envelope Estimation with error bars
        plt.errorbar(xk, uk, yerr=se_uk, label=rf'Estimate Moreau Envelope Value, $u(x_k,t_k))$, where $t_k={tk:.1f}$',
                 color='red', fmt='o', markersize=5, zorder=4, marker='x', capsize=5)

        # Plot Points for Current, Next, Initial Iteration, and Global Minima
        plt.scatter(xk, self.f(xk), label=r'Current Iteration, $f(x_k)$', zorder=6,s=150,  marker='^')
        plt.scatter(self.x0, self.f(self.x0), color='green', label=r'Initial Iteration, $f(x_0)$',s=100, zorder=4, facecolors='none',edgecolors='blue')
        plt.scatter(self.x_true, self.f(self.x_true), color='black', label=r'Global Minima, $f(x_{true})$',s=100, zorder=5, marker='x')
        #plt.scatter(xk_plus1, self.f(xk_plus1), color='cyan', zorder=4, label=r'Next Iteration, $f(x_{k+1})$', marker='^')

        plt.title(f'f(x) and Moreau Envelope\nIteration {k}, Error={error:.3e}, Tol={self.tol:.3e}')
        plt.xlabel('x')
        plt.ylabel('Function Value')

        # Set limits and grid
        plt.xlim(intervalx_a, intervalx_b)

        # Dynamically set the y-limits based on function outputs
        y_min = np.min(f_values)
        y_max = np.max(f_values)

        if y_max < 0:
            # If all values are negative, set the limits to give some visual space
            plt.ylim(1.2 * y_min, 0)  # Set lower limit 20% below min, upper limit at 0
        elif y_min > 0:
            plt.ylim(0.8 * y_min, 1.2 * y_max)  # 20% less than the min if min is positive
        else:
            plt.ylim(1.2 * y_min, 1.2 * y_max)  # 20% more than the min if min is zero or negative

        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1)
        #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=1)
        plt.grid(which='major', linestyle='-', linewidth='0.5')
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='minor', linestyle=':', linewidth='0.5')

        # Save the figure as a PNG file
        plt.savefig("MAD_interactive_plot.png", format='png', bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

        if plot_bool:
            with plot_output:
                clear_output(wait=True)  # Clear previous plot
                display(Image(filename="MAD_interactive_plot.png"))