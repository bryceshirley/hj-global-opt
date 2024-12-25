from math import pi

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, Checkbox, FloatText, HBox, Output, VBox
from matplotlib.ticker import MaxNLocator
from scipy.optimize import differential_evolution
from scipy.special import roots_hermite
from IPython.display import clear_output

from test_functions1D import (DiscontinuousFunc_numpy,
                              MultiMinimaAbsFunc_numpy, MultiMinimaFunc_numpy,
                              Sin_numpy, Sinc_numpy)

# h is the function to be minizer over z for parameters f, T, and x
def h(z, h_parameters):
    f, T, x = h_parameters # Unpack the parameters (f = selected_function, T = fixed_T)
    return f(z) + (1 / (2 * T)) * (z - x)**2

class HJMoreauAdaptiveDescentVisualizer:
    """
    A visualizer for the HJ Moreau Adaptive Descent algorithm.
    This class provides a graphical interface to visualize the Moreau envelope and the adaptive descent process.
    It allows users to select different functions, adjust parameters, and observe the behavior of the algorithm
    through plots and error histories.
    """
    def __init__(self):
        # Initialize global variables for function and plotting range
        self.selected_function = MultiMinimaFunc_numpy  # Default function
        self.intervalx_a, self.intervalx_b = -30.0, 30.0
        #self.global_min_x, self.global_min_f = -1.51035, -4.76275
        self.global_min_x = -1.51034568-10
        self.global_min_f = self.selected_function(self.global_min_x)
        self.value_T = 95
        self.value_rescale0 = 1 
        self.value_delta = 0.1
        self.value_int_samples = 1000
        self.gamma = 1.89777  # Default gamma for MultiMinima

        # Initialize widgets (replacing sliders with FloatText input fields)
        self.x_0_input = FloatText(value=5*(self.intervalx_b-self.intervalx_a)/6+self.intervalx_a, description='x_0:')
        self.fixed_T_input = FloatText(value=self.value_T, description='Fixed time, T:')
        self.delta_input = FloatText(value=self.value_delta, description='Delta:')
        self.int_samples_input = FloatText(value=self.value_int_samples, description='Samples, N:')
        self.rescale0_input = FloatText(value=self.value_rescale0, description='Rescale0:')
        self.plot_output = Output()
        self.error_plot_output = Output()
        self.t_threshold_display = HTML()
        self.alpha_input = FloatText(value=0.1, description='Alpha:')
        self.momentum_input = FloatText(value=0.5, description='Momentum:')
        self.spread_input = FloatText(value=1, description='Spread:')


        # Dropdowns
        self.function_dropdown = Dropdown(options=['MultiMinima', 'Sinc', 'Sin', 'MultiMinimaAbsFunc', 'DiscontinuousFunc'], value='MultiMinima', description='Function:')
        self.integration_dropdown = Dropdown(options=["HJ_GH", "HJ_MC","softmax_prox_MC","argmin_prox_MC"], value="HJ_MC", description='Integration:')
        self.distribution_dropdown = Dropdown(options=["Gaussian", "Laplace", "Cauchy"], value="Gaussian", description='Distribution:')
        self.integration_method = "HJ_MC"
        self.distribution = "Gaussian"


        # Checkbox for fixed time and acceleration
        self.scipy_moreau_envelope_checkbox = Checkbox(value=False, description='Scipy Moreau Envelope', tooltip='Check this to see the scipy moreau envelope.')
        self.sample_bool_checkbox = Checkbox(value=False, description='Display Samples', tooltip='Check to use Display Sampling.')

        # Create buttons
        self.iterate_button = Button(description='Iterate and Plot', button_style='info')
        self.run_200_iterations_button = Button(description='Run 200 Iterations', button_style='warning')
        self.reset_button = Button(description='Reset', button_style='success')

        # Bind Events Drop Downs
        self.function_dropdown.observe(self.function_dropdown_update, names='value')
        self.integration_dropdown.observe(self.integration_dropdown_update, names='value')
        self.distribution_dropdown.observe(self.distribution_dropdown_update, names='value')

        # Bind Events inputs
        self.x_0_input.observe(self.x_0_input_update, names='value')
        self.alpha_input.observe(self.alpha_input_update, names='value')
        self.momentum_input.observe(self.momentum_input_update, names='value')
        self.delta_input.observe(self.delta_input_update, names='value')
        self.rescale0_input.observe(self.rescale0_input_update, names='value')
        self.fixed_T_input.observe(self.fixed_T_input_update, names='value')
        self.int_samples_input.observe(self.int_samples_input_update, names='value')
        self.spread_input.observe(self.spread_input_update, names='value')

        # Bind Events Buttons
        self.iterate_button.on_click(self.run_iteration)
        self.reset_button.on_click(self.reset)
        self.run_200_iterations_button.on_click(self.run_200_iterations)

        # Bind Events Checkbox
        self.display_samples_bool = False
        self.display_scipy_ME_bool = False
        self.sample_bool_checkbox.observe(self.display_sampling, names='value')
        self.scipy_moreau_envelope_checkbox.observe(self.display_scipy_ME, names='value')

        # Display layout
        self.display_layout()

        # Initial t_threshold update
        self.update_t_threshold_display()

        # Initialize Error:
        self.initialize_history()

        # Update Moreau Envelope
        self.x_values = np.linspace(self.intervalx_a, self.intervalx_b, 100)
        self.update_moreau_envelope()

        self.update_plots()


    def display_layout(self):
        note = HTML("""<p>Select a function from the dropdown and adjust the inputs to modify the values of fixed time <strong>T</strong>, <strong>x_k</strong>, and Step Size <strong>&alpha;</strong>.</p>""")
        display(VBox([
            HTML('<h2>HJ Moreau Adaptive Descent Visualizer</h2>'),
            note,
            HBox([self.function_dropdown,self.integration_dropdown]),
            HBox([self.scipy_moreau_envelope_checkbox, self.sample_bool_checkbox, self.rescale0_input]),
            HBox([self.x_0_input, self.alpha_input, self.momentum_input]),
            HBox([self.fixed_T_input,self.int_samples_input,self.delta_input]),
            HBox([self.spread_input, self.distribution_dropdown]),
            self.t_threshold_display,  # Display for t_threshold
            HBox([self.iterate_button, self.run_200_iterations_button, self.reset_button]),  # Added plot button to the layout
            self.plot_output,
            self.error_plot_output
        ]))
        self.spread_input.layout.visibility = 'hidden'
        self.distribution_dropdown.layout.visibility = 'hidden'
    
    # ~~~~~~~~ Controller Methods: Buttons ~~~~~~~~~~

    def reset(self, _=None):
        self.initialize_history()
        self.update_t_threshold_display()
        self.update_plots()

    def run_200_iterations(self, button):
        """Run 200 iterations and update the plot."""
        for _ in range(30):
            self.update_x_k(button)  # Use the existing method to update x_k and history
        self.update_plots()  # Finally, update the plots after iterations

    def run_iteration(self, button):
        """Run 100 iterations and update the plot."""
        self.update_x_k(button)
        self.update_plots()

    #~~~~~~~~ Controller Methods: Drop Downs ~~~~~~~~~~

    def function_dropdown_update(self, change):
        self.update_function(change)
        self.initialize_history()
        self.update_t_threshold_display()
        self.update_plots()

    def integration_dropdown_update(self,change):
        self.integration_method = change['new']

        if self.integration_method == "argmin_prox_MC":
            self.delta_input.layout.visibility = 'hidden'
            self.rescale0_input.layout.visibility = 'hidden'
            self.spread_input.layout.visibility = 'visible'
            self.distribution_dropdown.layout.visibility = 'visible'
        elif self.integration_method == "softmax_prox_MC":
            self.spread_input.layout.visibility = 'visible'
            self.delta_input.layout.visibility = 'visible'
            self.rescale0_input.layout.visibility = 'visible'
            self.distribution_dropdown.layout.visibility = 'visible'
        else:
            self.delta_input.layout.visibility = 'visible'
            self.spread_input.layout.visibility = 'hidden'
            self.rescale0_input.layout.visibility = 'visible'
            self.distribution_dropdown.layout.visibility = 'hidden'

        self.initialize_history()
        self.update_plots()

    def distribution_dropdown_update(self,change):
        self.distribution = change['new']
        self.initialize_history()
        self.update_plots()

    def update_function(self, change):
        """Update the function and range based on dropdown selection."""
        if change['new'] == 'Sinc':
            self.min_T, self.max_T = 1, 1000
            self.selected_function = Sinc_numpy
            self.intervalx_a, self.intervalx_b = -20, 20
            self.gamma = 0.1259
        elif change['new'] == 'Sin':
            self.min_T, self.max_T = 1, 500
            self.selected_function = Sin_numpy
            self.intervalx_a, self.intervalx_b = -3.5 * pi, 2.5 * pi
            self.gamma = None
        elif change['new'] == 'MultiMinimaAbsFunc':
            self.min_T, self.max_T = 0.1, 100
            self.selected_function = MultiMinimaAbsFunc_numpy
            self.intervalx_a, self.intervalx_b = -15, 15
            self.gamma = 1.43457
        elif change['new'] == 'DiscontinuousFunc':
            self.min_T, self.max_T = 0.1, 100
            self.selected_function = DiscontinuousFunc_numpy
            self.intervalx_a, self.intervalx_b = -20, 15
            self.gamma = 7
        else:  # 'MultiMinima'
            self.min_T, self.max_T = 0.1, 100
            self.selected_function = MultiMinimaFunc_numpy
            self.intervalx_a, self.intervalx_b = -30, 30
            self.gamma = 1.89777

        # Compute global minimum
        result = differential_evolution(self.selected_function, [(self.intervalx_a, self.intervalx_b)])
        self.global_min_x = result.x
        self.global_min_f = result.fun

        #print(self.global_min_x)

        # Update inputs and plot
        self.x_0_input.value = (self.intervalx_a + self.intervalx_b) / 2
        self.fixed_T_input.value = (self.min_T + self.max_T) / 2

    #~~~~~~~~ Controller Methods: CheckBox ~~~~~~~~~~

    def display_sampling(self, change):
        """
        Update the sampling display state based on the checkbox.
        """
        self.display_samples_bool = change['new']
        self.update_plots()

    def display_scipy_ME(self, change):
        """
        Update the sampling display state based on the checkbox.
        """
        self.display_scipy_ME_bool = change['new']
        self.update_plots()

    #~~~~~~~~ Controller Methods: Inputs ~~~~~~~~~~

    def x_0_input_update(self, _):
        #self.initialize_history()
        #self.update_t_threshold_display()
        #self.update_plots()
        return

    def alpha_input_update(self, _):
        #self.initialize_history()
        #self.update_plots()
        return

    def momentum_input_update(self, _):
        #self.initialize_history()
        #self.update_plots()
        return

    def fixed_T_input_update(self, _):
        #self.initialize_history()
        #self.update_plots()
        return
    
    def rescale0_input_update(self, _):
        # self.initialize_history()
        # self.update_plots()
        return

    def delta_input_update(self, _):
        # self.initialize_history()
        self.update_plots()

    def int_samples_input_update(self, _):
        # self.initialize_history()
        # self.update_plots()
        return

    def spread_input_update(self, _):
        # self.initialize_history()
        # self.update_plots()
        return

    #~~~~~~~~ Display Updates ~~~~~~~~~~

    def update_t_threshold_display(self, extra=""):
        """Update the display for the t_threshold and compute its value."""

        # Create the HTML string
        html_content = """
        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <h4 style="margin: 0;">Time Threshold <i>(Theoretical Lower Bound on Initial Time Steps for Convergence)</i></h4>
        <div style="font-family: Times New Roman, serif; font-size: 14px;">
            <p><strong>t<sub>threshold</sub> = {}
        """

        # Calculate t_threshold
        if self.gamma is not None:
            t_threshold = (np.linalg.norm(self.global_min_x - self.x_0_input.value)**2) / (2 * self.gamma)
            # Use str.format to insert the calculated value into html_content
            self.t_threshold_display.value = html_content.format(f"""
                {t_threshold:.2f}, where</strong>:</p><p style="margin-left: 20px;">
                T ≥ ||x* - x<sub>k</sub>||<sup>2</sup> / (2γ) = t<sub>threshold</sub> > 0
                </p></div>""" + extra + "</div>")
        else:
            # If self.gamma is None, modify html_content to show 'UnDefined'
            self.t_threshold_display.value = html_content.format('UnDefined</p></div>')
            return

    #~~~~~~~~ Algorithm Methods ~~~~~~~~~~

    def scipy_compute_prox_and_grad(self, x_k, t= None):
        """
        Compute the prox and gradient using Scipy's differential evolution method.

        """
        if t is None:
            t = self.fixed_T_input.value
        int_samples = int(self.int_samples_input.value)

        # Find the global minimum using the differential evolution method
        hk_parameters = (self.selected_function, t, x_k)
        result = differential_evolution(h, [(self.intervalx_a, self.intervalx_b)], args=(hk_parameters,),popsize=int_samples)

        # Compute the correct gradient and prox
        prox_k = result.x[0]
        grad_k = (x_k - prox_k) / t

        return prox_k, grad_k
    
    def argmin_prox_MC(self, x_k,t=None):
        int_samples = int(self.int_samples_input.value)
        if t is None:
            t = self.fixed_T_input.value
        spread = self.spread_input.value

        if self.distribution == "Laplace":
            #Laplace distribution scale parameter b, related to fixed_T
            b = np.sqrt(t) / spread

            # Generate Laplace-distributed samples with mean x_k and scale b
            y = np.random.laplace(loc=x_k, scale=b, size=int_samples)

        elif self.distribution == "Cauchy":
            # Adjust this parameter for heavier tails if needed
            gamma = np.sqrt(t)/spread

            # Generate Cauchy-distributed samples with location x_k and scale gamma
            y = x_k + gamma * np.random.standard_cauchy(size=int_samples)
        else:
            standard_dev = np.sqrt(t)/spread

            # Generate Gaussian-distributed samples with mean x_k
            y =  standard_dev * np.random.randn(int_samples) + x_k


        h_parameters = (self.selected_function, t, x_k)

        # Prox Function:
        prox_function=h(y, h_parameters)

        prox_index = np.argmin(prox_function)

        prox_k = y[prox_index]

        grad_k = (x_k - prox_k)

        uk = prox_function[prox_index]

        return prox_k, grad_k, uk, y
    
    def softmax_prox_MC(self, x_k_input=None,t=None):
        """
        Compute the prox and gradient using the Hamilton-Jacobi (HJ) method.
        """
        if t is None:
            t = self.fixed_T_input.value

        delta = self.delta_input.value
        int_samples = int(self.int_samples_input.value)
        spread = self.spread_input.value

        if x_k_input is None:
            x_k = self.x_k
        else:
            x_k = x_k_input

        line_search_iterations=0
        
        if self.distribution == "Laplace":
            #Laplace distribution scale parameter b, related to fixed_T
            b = t

            # Generate Laplace-distributed samples with mean x_k and scale b
            y = np.random.laplace(loc=x_k, scale=b, size=int_samples)
        elif self.distribution == "Cauchy":
            # Adjust this parameter for heavier tails if needed
            gamma = t

            # Generate Cauchy-distributed samples with location x_k and scale gamma
            y = x_k + gamma * np.random.standard_cauchy(size=int_samples)
        else:
            standard_dev = np.sqrt(spread*delta*t)

            # Generate Gaussian-distributed samples with mean x_k and standard deviation
            y =  standard_dev * np.random.randn(int_samples) + x_k

        exponent = -(1/delta)*((self.selected_function(y))) #+ (1-delta/spread)*(x_k - y)**2 / (2 * t))
        maxscaled_exponent = exponent - np.max(exponent)
        
        exp_term = np.exp(maxscaled_exponent)
        
        w = exp_term / np.sum(exp_term)
        
        # Compute Numerator and average over the samples
        prox_k = np.dot(w, y)

        # Compute Gradient at uk (times tk)
        grad_k = (x_k - prox_k)

        # Compute Argmin of Samples
        h_parameters = (self.selected_function, t, x_k)
        prox_function=h(y, h_parameters)
        prox_index = np.argmin(prox_function)

        if x_k_input is None:
            #extra = f"\nf(x_k)): {self.selected_function(x_k)}, f(prox_k): {self.selected_function(prox_k)}, argminf(y_i): {self.selected_function(y[prox_index])}"
            current_prox_estimate = self.selected_function(x_k)+(1 / (2 * t)) * np.linalg.norm((x_k - self.x_0_input.value))**2
            new_prox_estimate = self.selected_function(prox_k)+(1 / (2 * t)) * np.linalg.norm((prox_k - self.x_0_input.value))**2
            argmin_prox_estimate = self.selected_function(y[prox_index])+(1 / (2 * t)) * np.linalg.norm((y[prox_index] - self.x_0_input.value))**2
            extra = f"\nCurrent Prox Estimate: {current_prox_estimate}, f(prox_k): {new_prox_estimate}, argminf(y_i): {argmin_prox_estimate}"

            if new_prox_estimate > current_prox_estimate:
                extra = "Reduce Delta"

            self.update_t_threshold_display(extra=extra)
        
        # Compute Moreau Envelope
        #v_delta = np.mean(np.exp(exponent))
        h_parameters = (self.selected_function, t, x_k)
        uk = h(prox_k, h_parameters)
        #uk = - delta * np.log(np.mean(np.exp(exponent))+ 1e-200)

        return prox_k, grad_k, uk, line_search_iterations, y
    
    
    def hj_compute_prox_and_grad_trap(self, x_k_input=None,t=None):
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
        if t is None:
            t = self.fixed_T_input.value

        delta = self.delta_input.value
        f = self.selected_function
        N = int(self.int_samples_input.value)

        if x_k_input is None:
            x_k = self.x_k
        else:
            x_k = x_k_input
        
        # Discretize the domain for trapezium rule
        a, b = x_k - 20 * np.sqrt(delta * t), x_k + 20 * np.sqrt(delta * t)  # Define an interval around x (5 standard deviations)
        y = np.linspace(a, b, N)
        width = (b - a) / (N - 1)

        # Remove max exponent to avoid overflow
        exponent = -f(y) / delta - (x_k - y)**2 / (2 * delta * t)
        exponent = exponent - np.max(exponent)

        # Calculate v_delta using the trapezium rule
        discrete_integrand_v_delta = np.exp(exponent)
        v_delta = (width / 2) * (discrete_integrand_v_delta[0] + 2 * np.sum(discrete_integrand_v_delta[1:N-1]) + discrete_integrand_v_delta[-1])
        
        # Calculate ∇v_delta using the trapezium rule
        discrete_integrand_grad_v_delta = y *discrete_integrand_v_delta
        grad_v_delta = (width / 2) * (discrete_integrand_grad_v_delta[0] + 2 * np.sum(discrete_integrand_grad_v_delta[1:N-1]) + discrete_integrand_grad_v_delta[-1])
        
        # Compute ∇u_delta using the simplified formula: -(grad_v_delta / v_delta)
        prox_k = grad_v_delta / (v_delta)

        # Compute prox_xk
        grad_k = (x_k - prox_k)

        # Compute estimated uk
        h_parameters = (self.selected_function, t, x_k)
        uk = h(prox_k, h_parameters)

        lineseach_iterations = 0

        # Return Gradient at uk, uk, prox at xk and standard error in uk sample
        return prox_k, grad_k, uk,lineseach_iterations, y

    def hj_compute_prox_and_grad_MC(self, x_k,t=None):
        """
        Compute the prox and gradient using the Hamilton-Jacobi (HJ) method.
        """
        if t is None:
            t = self.fixed_T_input.value

        delta = self.delta_input.value
        rescale = self.rescale0_input.value
        int_samples = int(self.int_samples_input.value)

        line_search_iterations=0

        while True:
            # Sample From a Guassian with mean x and standard deviation
            standard_dev = np.sqrt(delta * t/rescale)
            y = standard_dev * np.random.randn(int_samples) + x_k

            exponent = -rescale*self.selected_function(y) / delta
            maxscaled_exponent = exponent - np.max(exponent)
            
            exp_term = np.exp(maxscaled_exponent)
            
            w = exp_term / np.sum(exp_term)
        
            softmax_overflow = 1.0 - (w < np.inf).prod()
            if softmax_overflow:
                rescale /= 2
                line_search_iterations +=1
            else:
                break

         # Compute Numerator and average over the samples
        prox_k = np.dot(w, y)

        # Compute Gradient at uk (times tk)
        grad_k = (x_k - prox_k)

        # Compute Moreau Envelope
        #v_delta = np.mean(np.exp(exponent))
        h_parameters = (self.selected_function, t, x_k)
        uk = h(prox_k, h_parameters)

        return prox_k, grad_k, uk, line_search_iterations, y
    
    def hj_compute_prox_and_grad_GH(self, x_k,t=None):
        """
        Compute the prox and gradient using the Hamilton-Jacobi (HJ) method.
        """
        if t is None:
            t = self.fixed_T_input.value

        delta = self.delta_input.value
        rescale = self.rescale0_input.value
        int_samples = int(self.int_samples_input.value)

        z, weights = roots_hermite(int_samples)

        line_search_iterations=0

        while True:
            t_rescaled = t/rescale
            
            # Sample From a Guassian with mean x and standard deviation
            sigma = np.sqrt(2*delta*t_rescaled)

            y = x_k - z*sigma

            exponent = -rescale*self.selected_function(y) / delta

            maxscaled_exponent = exponent - np.max(exponent)
            
            exp_term = np.exp(maxscaled_exponent)
            
            # w = weights*exp_term / np.dot(weights, exp_term)
            w = np.divide(weights * exp_term, np.dot(weights, exp_term),
               out=np.full_like(weights, np.inf), where=np.dot(weights, exp_term) != 0)
        
            softmax_overflow = 1.0 - (w < np.inf).prod()
            if softmax_overflow:
                rescale /= 2
                line_search_iterations +=1
            else:
                break

        # Compute grad_k
        grad_k = sigma*np.dot(w, z)

        # Compute prox k
        prox_k = (x_k - grad_k)

        # Compute Moreau Envelope
        #v_delta = np.mean(np.exp(exponent))
        h_parameters = (self.selected_function, t, x_k)
        uk = h(prox_k, h_parameters)

        return prox_k, grad_k, uk, line_search_iterations, y

    def gradient_descent(self, x_k_input=None, method=None):
        """
        Perform standard gradient descent.

        Args:
            x_k: Current iterate value.
            method: Descent method ('HJ' or 'Scipy').

        Returns:
            x_k_plus_1: Updated iterate after gradient descent.
            grad_k: Gradient at x_k.
        """
        alpha = self.alpha_input.value

        if x_k_input is None:
            x_k = self.x_k
        else:
            x_k = x_k_input

        if method is None:
            method = self.integration_method

        if method == "HJ_MC":
            #prox_k, grad_k, _, _, _ = self.hj_compute_prox_and_grad_MC(x_k)
            prox_k, grad_k, _,_,_ = self.hj_compute_prox_and_grad_trap(x_k)
        elif method == "HJ_GH":
            prox_k, grad_k, _, _, _ = self.hj_compute_prox_and_grad_GH(x_k)
        elif method == "scipy":
            prox_k, grad_k = self.scipy_compute_prox_and_grad(x_k)
        elif method == "softmax_prox_MC":
            prox_k, grad_k, _, _, _ = self.softmax_prox_MC()
        elif method == "argmin_prox_MC":
            prox_k, grad_k, _, _ = self.argmin_prox_MC(x_k)
        else:
            raise ValueError(f"Unknown method {method}")

        # Perform gradient descent update
        x_k_plus_1 = x_k - alpha * (x_k-prox_k)

        if self.selected_function(x_k_plus_1) > self.selected_function(x_k):
            x_k_plus_1 = x_k

        return x_k_plus_1, grad_k
    
    # def accelerated_gradient_descent(self, momentum, method=None):
    #     """
    #     Perform accelerated gradient descent.
    #     """

    #     if method is None:
    #         method = self.integration_method

    #     if self.k > 1:
    #         y_k = self.acc_x_k + momentum * (self.acc_x_k - self.acc_x_k_minus_1)
    #         acc_x_k_plus_1, grad_k = self.gradient_descent(y_k, method=method)
    #     else:
    #         acc_x_k_plus_1, grad_k = self.gradient_descent(self.x_k, method=method)

    #     if self.selected_function(acc_x_k_plus_1) > self.selected_function(self.x_k):
    #         x_k_plus_1 = self.x_k

    #     return acc_x_k_plus_1, grad_k
    
    def initialize_history(self):
        # Initialize x_k for x_0
        self.k = 0
        self.x_k = self.x_0_input.value
        self.x_k_scipy = self.x_0_input.value
        # self.acc_x_k = self.x_0_input.value
        # self.acc_x_k_minus_1 = self.x_0_input.value

        # Compute f(x_0)
        f_k = self.selected_function(self.x_k)

        # Store the initial function value and initial errors
        # self.acc_f_k_hist = [f_k]
        # self.acc_error_k_hist = [np.linalg.norm(self.acc_x_k- self.global_min_x)]

        self.f_k_hist = [f_k]
        self.error_k_hist = [np.linalg.norm(self.x_k - self.global_min_x)]

        self.f_k_hist_scipy = [f_k]
        self.error_k_hist_scipy = [np.linalg.norm(self.x_k_scipy- self.global_min_x)]


        # _, grad_k = self.gradient_descent(self.x_k)
        # self.gk_hist = [(self.x_k,grad_k)]

    def update_x_k(self, button):
        """
        Update x_k and x_k_accelerated based on gradient descent and acceleration.

        Args:
            button: The trigger event (e.g., UI button click).
        """
        # momentum = self.momentum_input.value
        self.k += 1

        # Perform gradient descent for both methods
        x_k_plus_1, grad_k = self.gradient_descent(self.x_k)
        x_k_plus_1_scipy, _ = self.gradient_descent(self.x_k_scipy, method="scipy")

        # Update x_k for both methods
        self.x_k_scipy= x_k_plus_1_scipy
        self.x_k = x_k_plus_1

        # # Approximated Gradient
        # self.gk_hist.append((self.x_k, grad_k))


        # Perform accelerated gradient descent
        # acc_x_k_plus_1,_ = self.accelerated_gradient_descent(momentum)

        # Update accelerated x_k and x_k_minus
        # self.acc_x_k_minus_1 = self.acc_x_k
        # self.acc_x_k = acc_x_k_plus_1

        # Record function values and errors
        # Calculate the function value and error for the accelerated method
        # f_k_accelerated = self.selected_function(self.acc_x_k)
        # self.acc_f_k_hist.append(f_k_accelerated)
        # self.acc_error_k_hist.append(np.linalg.norm(self.acc_x_k - self.global_min_x))

        # Calculate the function value and error for the original method
        f_k = self.selected_function(self.x_k)
        self.f_k_hist.append(f_k)
        self.error_k_hist.append(np.linalg.norm(self.x_k - self.global_min_x))

        # Calculate the function value and error for the original method
        f_k_scipy = self.selected_function(self.x_k_scipy)
        self.f_k_hist_scipy.append(f_k_scipy)
        self.error_k_hist_scipy.append(np.linalg.norm(self.x_k_scipy- self.global_min_x))

    def update_plots(self):
        function_plt=self.update_function_plot()
        error_plt=self.update_plot_errors()

        # Clear previous plots
        self.error_plot_output.clear_output()
        self.plot_output.clear_output() 

        # Update plots 
        with self.error_plot_output:
            error_plt.show()
        with self.plot_output:
            function_plt.show()

    def update_moreau_envelope(self, method=None,t=None):
        if t is None:
            t = self.fixed_T_input.value

        if method is None:
            method = self.integration_method

        # Compute Moreau Envelope
        if method == "HJ_MC":
            self.u_values = []

            avg_line_search_iterations = 0

            for xk in self.x_values:
                _, _, uk, line_search_iterations, _ = self.hj_compute_prox_and_grad_trap(xk,t=t)
                avg_line_search_iterations += line_search_iterations
                self.u_values.append(uk)
            avg_line_search_iterations /= len(self.x_values)

            #print(f"Avg Line Search Iterations: {avg_line_search_iterations}")
        elif method == "HJ_GH":
            self.u_values = []

            avg_line_search_iterations = 0

            for xk in self.x_values:
                _, _, uk, line_search_iterations, _ = self.hj_compute_prox_and_grad_GH(xk,t=t)
                avg_line_search_iterations += line_search_iterations
                self.u_values.append(uk)
            avg_line_search_iterations /= len(self.x_values)

            #print(f"Avg Line Search Iterations: {avg_line_search_iterations}")
        elif method == "softmax_prox_MC":
            self.u_values = []

            avg_line_search_iterations = 0

            for xk in self.x_values:
                _, _, uk, line_search_iterations, _ = self.softmax_prox_MC(xk,t=t)
                avg_line_search_iterations += line_search_iterations
                self.u_values.append(uk)
            avg_line_search_iterations /= len(self.x_values)

            #print(f"Avg Line Search Iterations: {avg_line_search_iterations}")
        elif method == "argmin_prox_MC":
            self.u_values = []

            avg_line_search_iterations = 0

            for xk in self.x_values:
                _, _, uk, _ = self.argmin_prox_MC(xk,t=t)
                self.u_values.append(uk)

        elif method == "scipy":
            self.u_values = [differential_evolution(h, [(self.intervalx_a, self.intervalx_b)], args=((self.selected_function, t, x),)).fun for x in self.x_values]
        else:
            raise ValueError(f"Unknown method {method}")

    def fit_quadratic_to_moreau_envelope(self):

        # Set up the design matrix for quadratic fitting
        A = np.vstack([self.x_values**2, self.x_values, np.ones_like(self.x_values)]).T

        # Fit the quadratic polynomial (degree 2) using least squares
        coeffs, _, _, _ = np.linalg.lstsq(A, self.u_values, rcond=None)

        # Extract coefficients
        return coeffs

    def update_function_plot(self):
        """Update the plot based on slider values."""
        fixed_T = self.fixed_T_input.value
        alpha = self.alpha_input.value

        # Compute prox at x_k
        if self.integration_method == "HJ_GH":
            x_hat, _, _, _, samples  = self.hj_compute_prox_and_grad_GH(self.x_k)
        elif self.integration_method == "HJ_MC":
            x_hat, _, _, _, samples  = self.hj_compute_prox_and_grad_trap(self.x_k)
        elif self.integration_method == "softmax_prox_MC":    
            x_hat, _, _, _, samples  = self.softmax_prox_MC(self.x_k)
        elif self.integration_method == "argmin_prox_MC":    
            x_hat, _, _, samples  = self.argmin_prox_MC(self.x_k)

        hk_parameters = (self.selected_function, fixed_T, self.x_k)

        f_hat = h(x_hat, hk_parameters)

        f_values = self.selected_function(self.x_values)
        h_values = h(self.x_values, hk_parameters)

        # Compute next iteration
        x_k_plus_1 = self.x_k- alpha * (self.x_k- x_hat)
        f_x_k_plus_1 = self.selected_function(x_k_plus_1)

        # Display Samples
        plt.figure(figsize=(10, 6))
        if self.display_samples_bool:
            samples_func_values = np.array([self.selected_function(sample) for sample in samples])# + (1 / (2 * tk)) * (sample - xk) ** 2
            plt.scatter(samples, samples_func_values,color='blue', label=r'Samples', zorder=4, marker='*')

        # Plotting f(x)
        plt.plot(self.x_values, f_values, label=r'$f(x)$', color='black')
        plt.plot(self.x_values, h_values, label=r'Function to minimize at $x_k$, $f(z) + \frac{1}{2t} ||z - x_k||^2$', color='orange')

        # Plot Moreau Envelope u(x)
        self.update_moreau_envelope()
        plt.plot(self.x_values, self.u_values, label=r'HJ Moreau Envelope, $u^{\delta}(x,T)$', color='green')
        plt.scatter(self.x_k, f_hat, label=r'Moreau Envelope at $x_k$, $u(x_k,T)$', color='red', marker='x')

        # Calculate the value of f at x_k
        f_xk_value = self.selected_function(self.x_k)
        plt.scatter(self.x_k, f_xk_value, color='black', zorder=5, label=r'Current Iteration $f(x_k)$', marker='x')

        if self.display_scipy_ME_bool:
            self.update_moreau_envelope(method="scipy")
            plt.plot(self.x_values, self.u_values, label=r'Scipy Moreau Envelope, $u(x,t)$', color='blue')
            # x_hat_scipy, _ = self.scipy_compute_prox_and_grad(self.x_k)
            # f_hat_scipy = h(x_hat_scipy, hk_parameters)
            # plt.scatter(x_hat_scipy, f_hat_scipy, color='red', zorder=5, label=r'$prox_{tf}(x_k)$')
            # plt.plot([x_hat_scipy, self.x_k], [f_hat_scipy, f_hat_scipy], color='red', linewidth=0.8, linestyle=':')#,label='Line between ' + r' $x_k$ '+f',\n'+r'$prox(x_k)$ and envelope $u(x_k,T)$')
            # plt.plot([self.x_k, self.x_k], [f_xk_value, f_hat_scipy], color='red', linewidth=0.8, linestyle=':')

        # Mark the Prox at x_k
        plt.scatter(x_hat, f_hat, color='red', zorder=5, label=r'Proximal at x_k, $\hat{x}_k=prox_{Tf}(x_k)$')

        # Accelerated xk
        #acc_f_xk_value = self.selected_function(self.acc_x_k)
        #plt.scatter(self.acc_x_k, acc_f_xk_value, color='blue', zorder=5, label=r'Current Iteration Accelerated $f(x_k)$', marker='x')

        # Mark the global minimum of f(x)
        plt.scatter(self.global_min_x, self.global_min_f, color='cyan', zorder=5, label=r'Global Minima, $f(x_{true})$', marker='o')

        # Plot the point for f(x_{k+1})
        # plt.scatter(x_k_plus_1, f_x_k_plus_1, color='magenta', zorder=5, 
        #             label=r'Next Iteration $f(x_{k+1})$, for $x_{k+1}=x_k-\alpha T\nabla u(x_k)$', marker='^')
        #acc_x_k_plus_1,grad_k = self.accelerated_gradient_descent(self.momentum_input.value)

        #acc_f_xk_value = self.selected_function(acc_x_k_plus_1)
        #plt.scatter(acc_x_k_plus_1, acc_f_xk_value, color='cyan', zorder=5, 
        #            label=r'Next Iteration Accelerated $f(x_{k+1})$', marker='^')

        plt.plot([x_hat, self.x_k], [f_hat, f_hat], color='red', linewidth=0.8, linestyle=':',label='Line between ' + r' $x_k$ '+f',\n'+r'$prox(x_k)$ and envelope $u(x_k,T)$')
        plt.plot([self.x_k, self.x_k], [f_xk_value, f_hat], color='red', linewidth=0.8, linestyle=':')
            
        # Add titles and labels
        plt.title(f'Visualization of Moreau Envelope for f(x) Using Monte Carlo,\n T={fixed_T:.2f},delta={self.delta_input.value:.3f},N={self.int_samples_input.value}')#\n Sampling: {self.integration_method}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5, ls='--')
        plt.axvline(0, color='black', linewidth=0.5, ls='--')

        # Set limits and grid
        plt.xlim(self.intervalx_a, self.intervalx_b)

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

        # plt.grid(which='major', linestyle='-', linewidth='0.5')
        # plt.minorticks_on()
        # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        # plt.grid(which='minor', linestyle=':', linewidth='0.5')
        plt.legend(loc='best')

        return plt
    
    def plot_time_evolution(self):
        """Update the plot based on slider values."""
        plt.figure(figsize=(10, 6))
        f_values = self.selected_function(self.x_values)

        # Plotting f(x)
        plt.plot(self.x_values, f_values, label=r'Intial Condition',color='black')

        # Mark the global minimum of f(x)
        plt.scatter(self.global_min_x, self.global_min_f, color='black', zorder=5, label=r'Global Minima, $f(x_{true})$', marker='x')

        for t in [0.1, 5, 30, 100]:
            # Plot Moreau Envelope u(x)
            self.update_moreau_envelope(t=t)
            plt.plot(self.x_values, self.u_values, linestyle='--', label=f'HJ-MC Viscous Moreau Envelope, t={t}', linewidth='1.5')

            # self.update_moreau_envelope(method="scipy",t=t)
            # plt.plot(self.x_values, self.u_values, label=f'Scipy Viscous Moreau Envelope, t={t}')

            
        # Add titles and labels
        plt.title(f'Time Evolution of Estimate Moreau Envelope\ndelta={self.delta_input.value:.3f},N={self.int_samples_input.value}\n Sampling: {self.integration_method}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5, ls='--')
        plt.axvline(0, color='black', linewidth=0.5, ls='--')

        # Set limits and grid
        plt.xlim(self.intervalx_a, self.intervalx_b)

        # Dynamically set the y-limits based on function outputs
        y_min = np.min(f_values)
        y_max = np.max(f_values)

        if y_max < 0:
            # If all values are negative, set the limits to give some visual space
            plt.ylim(1.5 * y_min, 0)  # Set lower limit 20% below min, upper limit at 0
        elif y_min > 0:
            plt.ylim(0.5 * y_min, 1.2 * y_max)  # 20% less than the min if min is positive
        else:
            plt.ylim(1.5 * y_min, 1.2 * y_max)  # 20% more than the min if min is zero or negative

        # plt.grid(which='major', linestyle='-', linewidth='0.5')
        # plt.minorticks_on()
        # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        # plt.grid(which='minor', linestyle=':', linewidth='0.5')
        plt.legend(loc='best')

        plt.show

    def update_plot_errors(self):

        plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

        # First subplot for error history
        plt.subplot(1, 2, 1)
        if self.display_scipy_ME_bool:
            plt.semilogy(self.error_k_hist_scipy, marker='*', linestyle='-', label='Gradient Descent SCIPY', color='green')
        #plt.semilogy(self.acc_error_k_hist, marker='o', linestyle='-', label='Accelerated Gradient Descent', color='blue')
        plt.semilogy(self.error_k_hist, marker='^', linestyle='-', label='Gradient Descent', color='red')
        plt.title('Error History')
        plt.xlabel('Iteration')  # Add x-axis label here
        plt.legend()  # Show legend

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))

        # Second subplot for f_k history
        plt.subplot(1, 2, 2)
        if self.display_scipy_ME_bool:
            plt.plot(self.f_k_hist_scipy, marker='*', linestyle='-', label='Gradient Descent SCIPY', color='green')
        #plt.plot(self.acc_f_k_hist, marker='o', linestyle='-', label='Accelerated Method Gradient Descent', color='blue')
        plt.plot(self.f_k_hist, marker='^', linestyle='-', label='Gradient Descent', color='red')  
        plt.title('f_k History')
        plt.xlabel('Iteration')  # Add x-axis label here
        plt.legend()  # Show legend

        # Set the x-axis to start from an integer
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower'))

        return plt