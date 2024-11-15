# UI Class to Interact with HJ_MAD



from math import pi

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, clear_output, display
from ipywidgets import (HTML, Button, Checkbox, Dropdown, FloatSlider,
                        FloatText, HBox, Output, VBox)

from test_functions1D import (DiscontinuousFunc_numpy,
                              MultiMinimaAbsFunc_numpy, MultiMinimaFunc_numpy,
                              Sin_numpy, Sinc_numpy)

from hj_mad_1d_numpy import HJ_MAD_1D_NUMPY

class HJ_MAD_1D_UI:
    """
    A user interface class for visualizing and interacting with the Hamilton-Jacobi Moreau Adaptive Descent (HJ-MAD) algorithm in 1D.

    This class provides a GUI to set up, run, and visualize the HJ-MAD optimization process. Users can adjust parameters, select functions to optimize, and view results through interactive plots.

    Attributes:
        plot_output (Output): Widget to display the plot.
        plot_results_output (Output): Widget to display the results plot.
        t_min (float): Minimum time for the optimization.
        gamma (float): Parameter for time threshold calculation.
        hj_mad (HJ_MAD_1D_NUMPY): Instance for running the optimization.

    Methods:
        __init__(): Initializes the class and creates the UI.
        init_hj_mad(): Initializes the HJ_MAD_1D_NUMPY instance with default parameters.
        create_ui(): Creates the UI with sliders, checkboxes, and buttons.
        update_hj_mad_delta(b): Updates the delta parameter.
        update_hj_mad_int_samples(b): Updates the number of samples.
        update_hj_mad_t_init(b): Updates the initial time.
        update_hj_mad_t_max(b): Updates the maximum time.
        update_hj_mad_max_iters(b): Updates the maximum number of iterations.
        update_hj_mad_alpha(b): Updates the step size (alpha).
        update_hj_mad_beta(b): Updates the beta parameter for Adam method.
        update_hj_mad_momentum(b): Updates the momentum parameter.
        update_t_threshold_display(): Updates the time threshold display.
        update_standard_error_display(): Updates the standard error display.
        update_hj_mad_x0(b): Updates the initial position (x0).
        update_function(change): Updates the selected function based on the dropdown.
        update_fixed_time(change): Updates the fixed time state.
        update_acceleration(change): Updates the accelerated state.
        update_integration_method_visibility(change): Updates the integration method visibility.
        display_sampling(change): Updates the sampling display state.
        run(animate): Runs the HJ-MAD optimization process.
        plot_hj_mad(): Plots the initial state of the optimization.
        plot_results(algorithm_hist): Plots the optimization results.
        reset_plot(): Resets the plot to the initial state.
        tol_input_update(_): Updates the tolerance parameter.
        print_slider_values(): Prints the current slider values and selected function.
    """
    def __init__(self):
        self.init_hj_mad()
        self.create_ui()

    def init_hj_mad(self):
        # Output widget to display the plot
        self.plot_output = Output()
        self.plot_results_output = Output()

        # Fixed Variables
        tol = 5e-4
        eta_min = 0.9
        eta_plus = 5.0
        eta_vec = [eta_min, eta_plus]
        theta = 0.9
        beta = 0.0
        self.t_min = 0.1

        # Set Initial Interactive Values
        int_samples = int(200)
        delta = 0.1
        t_init = 220
        t_max = 500
        max_iters = int(50)
        alpha = 0.1

        # Default Function Settings
        intervalx_a, intervalx_b = -40, 25
        plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
        f = MultiMinimaFunc_numpy
        x_true = -1.51034568-10
        self.gamma = 1.89777
        x0 = -40

        # Create HJ_MAD instance with initial parameters
        self.hj_mad = HJ_MAD_1D_NUMPY(f=f, x_true=x_true, x0=x0, delta=delta, int_samples=int_samples, 
                             t_vec=[t_init, self.t_min, t_max], max_iters=max_iters, tol=tol, 
                             theta=theta, beta=beta, eta_vec=eta_vec, alpha=alpha, 
                             fixed_time=False, verbose=False, accelerated=False,
                             plot_parameters=plot_parameters)
    
    def create_ui(self):
        # Title and Section Headers:
        title = HTML("<h2>HJ Moreau Adaptive Descent Visualization (in 1D)</h2>")
        internal_parameters = HTML("<h3>Internal Parameters:</h3>")
        initial_conditions = HTML("<h3>Initial Conditions Parameters:</h3>")
        
        # Dropdown to select the function
        self.function_dropdown = Dropdown(
            options=['Sinc', 'Sin', 'MultiMinima','MultiMinimaAbsFunc', 'DiscontinuousFunc'],
            value='MultiMinima',
            description='Function:',
            tooltip='Select the mathematical function to optimize.'
        )
        
        # Sliders for parameters, initialized with default values
        self.x_0_slider = FloatSlider(value=self.hj_mad.x0, min=self.hj_mad.plot_parameters[0], max=self.hj_mad.plot_parameters[1], step=0.01, description=r'x_0:', tooltip='Adjust the initial position.')
        self.delta_slider = FloatSlider(value=self.hj_mad.delta, min=0, max=5.0, step=0.005, description='Viscosity, δ:', tooltip='Adjust the viscosity parameter.')
        self.int_samples_slider = FloatText(value=self.hj_mad.int_samples, description='Samples, n:', tooltip='Select the number of samples.')
        self.max_iters_slider = FloatSlider(value=self.hj_mad.max_iters, min=1, max=500, step=1, description='Max Iter., N:', tooltip='Set the maximum number of iterations.')
        self.alpha_slider = FloatSlider(value=self.hj_mad.alpha, min=1-np.sqrt(self.hj_mad.eta_vec[0]), max=1+np.sqrt(self.hj_mad.eta_vec[1]), step=0.01, description='Step Size, α:', tooltip='Adjust the step size for optimization.')
        self.momentum_slider = FloatSlider(value=self.hj_mad.momentum, min=0.01, max=2, step=0.01, description='Momentum:', tooltip='Adjust the momentum for accelerate GD.')

        self.t_max_slider = FloatSlider(value=self.hj_mad.t_vec[2], min=self.t_min, max=300, step=0.1, description='Max t, T:', tooltip='Set the maximum time for the optimization.')
        self.t_init_slider = FloatSlider(value=self.hj_mad.t_vec[0], min=self.t_min, max=300, step=0.1, description='Initial t, t₀:', tooltip='Set the initial time for the optimization.')

        # Text input
        self.beta_input = FloatText(value=self.hj_mad.beta, description='ADAM, β:', tooltip='Adjust the Adam parameter for Adam method.')
        
        # Only visible when acceleration is on
        self.momentum_slider.layout.visibility = 'hidden'
        self.beta_input.layout.visibility = 'hidden'

        # Value input for tolerence
        self.tol_input = FloatText(value=self.hj_mad.tol, description='tol:', tooltip='Set the tolerence for optimization.')
        self.plot_hj_mad()

        # Checkbox for fixed time and acceleration
        self.fixed_time_checkbox = Checkbox(value=False, description='Fixed Time', tooltip='Check to use fixed time for optimization.')
        self.acceleration_checkbox = Checkbox(value=False, description='Acceleration', tooltip='Check to use Accelerated GD for optimization.')
        self.sample_bool_checkbox = Checkbox(value=False, description='Display Samples', tooltip='Check to use Display Sampling.')

        # Checkbox to select the integration method
        self.integration_method_checkbox = Checkbox(
            value=False,
            description='Use Trapezium Rule Integration',
            tooltip='Check to use the trapezium rule instead of Monte Carlo sampling for integration.'
        )

        # Buttons to run optimization
        self.run_button = Button(description='Run Optimization', tooltip='Start the optimization process.')
        self.animation_button = Button(description='Run Animation', tooltip='Start the animation of the optimization process.')
        self.reset_plot_button = Button(description='Reset Plot', tooltip='Reset the plot to the initial state.')

        # Bind slider changes to specific update methods
        self.delta_slider.observe(self.update_hj_mad_delta, names='value')
        self.int_samples_slider.observe(self.update_hj_mad_int_samples, names='value')
        self.t_init_slider.observe(self.update_hj_mad_t_init, names='value')
        self.t_max_slider.observe(self.update_hj_mad_t_max, names='value')
        self.max_iters_slider.observe(self.update_hj_mad_max_iters, names='value')
        self.alpha_slider.observe(self.update_hj_mad_alpha, names='value')
        self.momentum_slider.observe(self.update_hj_mad_momentum, names='value')
        self.x_0_slider.observe(self.update_hj_mad_x0, names='value')

        # Bind tolerence input changes to specific update methods
        self.tol_input.observe(self.tol_input_update, names='value')
        self.beta_input.observe(self.update_hj_mad_beta, names='value')

        # Dropdown event listener to update the selected function
        self.function_dropdown.observe(self.update_function, names='value')

        # Checkbox event listener to update the fixed time state
        self.fixed_time_checkbox.observe(self.update_fixed_time, names='value')
        self.acceleration_checkbox.observe(self.update_acceleration, names='value')
        self.sample_bool_checkbox.observe(self.display_sampling, names='value')
        self.integration_method_checkbox.observe(self.update_integration_method_visibility, names='value')

        
        # # Initialize t_threshold_display and standard_error_display
        self.t_threshold_display = HTML("")
        self.standard_error_display = HTML("")
        self.update_t_threshold_display()
        self.update_standard_error_display()

        # Bind button clicks to the respective methods
        self.run_button.on_click(lambda b: self.run(animate=False))
        self.animation_button.on_click(lambda b: self.run(animate=True))
        self.reset_plot_button.on_click(lambda b: self.reset_plot())

        # Display the UI
        ui = VBox([title, internal_parameters, 
                HBox([self.function_dropdown, self.tol_input,self.max_iters_slider]),
                HBox([self.delta_slider, self.int_samples_slider, self.t_max_slider]), 
                HBox([self.alpha_slider,self.momentum_slider,self.beta_input]),
                HBox([self.sample_bool_checkbox, self.fixed_time_checkbox, self.acceleration_checkbox]),
                self.integration_method_checkbox,
                initial_conditions, 
                HBox([self.t_init_slider, self.x_0_slider]),
                HBox([self.standard_error_display, self.t_threshold_display]),
                HBox([self.run_button, self.animation_button,self.reset_plot_button]), # instructions,
                HBox([self.plot_output,self.plot_results_output])])  # Add plot output to the UI

        display(ui)
        

    def update_hj_mad_delta(self,b):
        self.hj_mad.delta = self.delta_slider.value
        self.plot_hj_mad()
        self.update_standard_error_display()

    def update_hj_mad_int_samples(self,b):
        self.hj_mad.int_samples = int(self.int_samples_slider.value)
        self.plot_hj_mad()
        self.update_standard_error_display()

    def update_hj_mad_t_init(self,b):
        self.hj_mad.t_vec[0] = self.t_init_slider.value

        # Don't let t_max become smaller than t_init
        if self.t_max_slider.value < self.t_init_slider.value:
            self.t_max_slider.value = self.t_init_slider.value
        self.plot_hj_mad()
        self.update_standard_error_display()

    def update_hj_mad_t_max(self,b):
        self.hj_mad.t_vec[2] = self.t_max_slider.value

        # Don't let t_max become smaller than t_init
        if self.t_max_slider.value < self.t_init_slider.value:
            self.t_max_slider.value = self.t_init_slider.value
        self.update_standard_error_display()

    def update_hj_mad_max_iters(self,b):
        self.hj_mad.max_iters = int(self.max_iters_slider.value)

    def update_hj_mad_alpha(self,b):
        self.hj_mad.alpha = self.alpha_slider.value

    def update_hj_mad_beta(self,b):
        self.hj_mad.beta = self.beta_input.value

    def update_hj_mad_momentum(self,b):
        self.hj_mad.momentum = self.momentum_slider.value

    def update_t_threshold_display(self):
        """Update the display for the t_threshold and compute its value."""

        # Create the HTML string
        html_content = """
        <div style="border: 1px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <h4 style="margin: 0;">Time Threshold <i>(Theoretical Lower Bound on Intial Time Steps for Convergence)</i></h4>
        <div style="font-family: Times New Roman, serif; font-size: 14px;">
            <p><strong>t<sub>threshold</sub> = {}
        """

        # Calculate t_threshold
        if self.gamma is not None:
            self.t_threshold = (np.linalg.norm(self.hj_mad.x_true - self.x_0_slider.value)**2) / (2 * self.gamma)
            # Use str.format to insert the calculated value into html_content
            html_content = html_content.format(f"{self.t_threshold:.2f}, where</strong>:</p>")
        else:
            # If self.gamma is None, modify html_content to show 'UnDefined'
            self.t_threshold_display.value = html_content.format('UnDefined</p>')
            return

        # Make sure t_min is less that t_threshold (this isn't a necessary 
        # condition as long as t_min<=t_init but this gives the user more choice)
        if self.t_threshold < self.t_min:
            self.t_min = self.t_threshold*0.8
            self.t_max_slider.min = self.t_min
            self.t_max_slider.min = self.t_min


        if self.t_threshold > self.t_max_slider.max:
            self.t_max_slider.max = self.t_threshold*1.5
            self.t_init_slider.max = self.t_threshold*1.5


        if self.hj_mad.fixed_time:
            html_content += f"""<p style="margin-left: 20px;">
                • t<sub>0</sub> ≥ ||x* - x<sub>0</sub>||<sup>2</sup> / (2γ) = t<sub>threshold</sub>
            </p>
            </div>
            </div>"""
            # Update the display
            self.t_threshold_display.value = html_content
            return
        
        html_content += f""" <p style="margin-left: 20px;">
                • T ≥ t<sub>0</sub> ≥ ||x* - x<sub>0</sub>||<sup>2</sup> / (2γ) = t<sub>threshold</sub>
            </p>
            <p style="margin-left: 20px;">
                • Also T ≥ t<sub>0</sub> ≥ τ > 0, τ = {self.t_min:.2f}
            </p>
            <p style="margin-left: 20px;">• T, max time. τ, min time. t<sub>0</sub>, initial time. </p>
        </div>
        </div>
        """
        # Update the display
        self.t_threshold_display.value = html_content


    def update_standard_error_display(self):
        """
        Standard Error is a measure of how much the sample mean will vary from 
        sample to sample.
        """
        # Compute the standard Deviation and Standard Error at Initial Time
        uk_info = self.hj_mad.compute_grad_uk(self.x_0_slider.value, self.t_init_slider.value)
        se_uk=uk_info[2]

        self.standard_error_display.value = f"""
            <div style="border: 1px solid black; padding: 10px; border-radius: 5px; margin-top: 10px;">
                <h4 style="margin: 0;">Standard Error for u<sub>k</sub> <i>(Variation Of The Sample Mean Between Samples)</i></h4>
            <div style="font-family: Times New Roman, serif; font-size: 14px;">
                <p><strong>Standard Error = <sup>σ</sup>/<sub>/√n</sub> = {se_uk:.3f}, where</strong>:</p>
                    <p style="margin-left: 20px;">• SE is Standard Error.</p>
                    <p style="margin-left: 20px;">• σ is the Standard Deviation(depends on δ, t, f and x).</p>
                    <p style="margin-left: 20px;">• n is the sample size.</p>
            </div>
            </div>
            """

    def update_hj_mad_x0(self,b):
        self.hj_mad.x0 = self.x_0_slider.value
        self.update_t_threshold_display()
        self.plot_hj_mad()

    def update_function(self, change):
        # Update the selected function based on dropdown selection
        if change['new'] == 'Sinc':
            intervalx_a, intervalx_b = -20, 20
            self.hj_mad.plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
            self.hj_mad.f = Sinc_numpy
            self.hj_mad.x_true = 4.49336839, # and -4.49336839

            self.gamma = 0.1259
        elif change['new'] == 'Sin':
            intervalx_a, intervalx_b = -3.5 * pi, 2.5 * pi
            self.hj_mad.plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
            self.hj_mad.f = Sin_numpy
            self.hj_mad.x_true = -np.pi / 2
            self.gamma = None
        elif change['new'] == 'MultiMinimaAbsFunc':
            intervalx_a, intervalx_b = -15, 15
            self.hj_mad.plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
            self.hj_mad.f = MultiMinimaAbsFunc_numpy
            self.hj_mad.x_true = 2
            self.gamma = 1.43457
        elif change['new'] == 'DiscontinuousFunc':
            intervalx_a, intervalx_b = -15, 15
            self.hj_mad.plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
            self.hj_mad.f = DiscontinuousFunc_numpy
            self.hj_mad.x_true = 2
            self.gamma = 7
        else:  # change['new'] == 'MultiMinimaFunc'
            intervalx_a, intervalx_b = -40, 25
            self.hj_mad.plot_parameters = [intervalx_a, intervalx_b,self.plot_output]
            self.hj_mad.f = MultiMinimaFunc_numpy
            self.hj_mad.x_true = -1.51034568-10
            self.gamma = 1.89777


        # Update the slider range for x_0 based on the new function
        self.x_0_slider.min = intervalx_a
        self.x_0_slider.max = intervalx_b
        self.x_0_slider.value = (intervalx_a + intervalx_b) / 2

        self.update_t_threshold_display()
        self.update_standard_error_display()
        self.plot_results_output.clear_output()  # Clear previous results
        self.plot_hj_mad()
    
    def update_fixed_time(self, change):
        """
        Update the fixed time state based on the checkbox.
        """
        self.hj_mad.fixed_time = change['new']
        # self.update_standard_error_display()
        # self.update_t_threshold_display()
        if change['new']:
            self.t_max_slider.layout.visibility = 'hidden'
        else:
            self.t_max_slider.layout.visibility = 'visible'

    def update_acceleration(self, change):
        """
        Update the accelerated state based on the checkbox.
        """
        self.hj_mad.accelerated = change['new']
        if change['new']:  # If the checkbox is checked
            self.momentum_slider.layout.visibility = 'visible'
            if not self.hj_mad.trap_integration:
                self.beta_input.layout.visibility = 'visible'
                self.beta_input.value = 0.1
            else:
                self.beta_input.value = 0.0
        else:  # If the checkbox is unchecked
            self.momentum_slider.layout.visibility = 'hidden'
            self.beta_input.layout.visibility = 'hidden'
            self.beta_input.value = 0.0

    def update_integration_method_visibility(self, change):
        # Show `beta` input only if the trapezium rule is NOT selected
        self.hj_mad.trap_integration = change['new']
        if change['new']:
            self.beta_input.layout.visibility = 'hidden'
            self.sample_bool_checkbox.layout.visibility = 'hidden'
            self.standard_error_display.layout.visibility = 'hidden'
            self.beta_input.value = 0.0
            self.sample_bool_checkbox.value = False
            self.display_sampling(self, change)
        elif self.hj_mad.accelerated:
            self.beta_input.layout.visibility = 'visible'
            self.sample_bool_checkbox.layout.visibility = 'visible'
            self.standard_error_display.layout.visibility = 'visible'
        else:
            self.sample_bool_checkbox.layout.visibility = 'visible'
            self.standard_error_display.layout.visibility = 'visible'

    def display_sampling(self, change):
        """
        Update the sampling display state based on the checkbox.
        """
        self.hj_mad.sample_bool = change['new']
        self.plot_results_output.clear_output()  # Clear previous results
        self.plot_hj_mad()

    def run(self, animate):
        self.plot_results_output.clear_output()  # Clear previous results
        # Run optimization with current slider values
        print('-------------------------- RUNNING HJ-MAD ---------------------------')
        print('For the parameters:')
        self.print_slider_values()

        _, algorithm_hist = self.hj_mad.run(animate=animate)
        
        self.plot_results(algorithm_hist)
        display(Image(filename="MAD_interactive_plot.png"))

    def plot_hj_mad(self):
        self.hj_mad.plot(0, self.hj_mad.x0, self.hj_mad.t_vec[0],0)
    
    def plot_results(self, algorithm_hist):
        # Unpack general case and accelerated case histories
        non_acc_algorithm_hist = None

        if self.hj_mad.accelerated:
            self.hj_mad.accelerated = False
            _, non_acc_algorithm_hist = self.hj_mad.run(animate=False,plot_bool=False)
            _, non_acc_tk_hist, non_acc_xk_error_hist, non_acc_rel_grad_uk_norm_hist, non_acc_fk_hist = non_acc_algorithm_hist

            if self.hj_mad.trap_integration:
                self.hj_mad.accelerated = True
            else:
                self.hj_mad.beta = 0.7
                _, adam_algorithm_hist = self.hj_mad.run(animate=False,plot_bool=False)
                _, adam_tk_hist, adam_xk_error_hist, adam_rel_grad_uk_norm_hist, adam_fk_hist = adam_algorithm_hist

                self.hj_mad.accelerated = True

                self.hj_mad.beta = 0.3
                _, acc_adam_algorithm_hist = self.hj_mad.run(animate=False,plot_bool=False)
                _, acc_adam_tk_hist, acc_adam_xk_error_hist, acc_adam_rel_grad_uk_norm_hist, acc_adam_fk_hist = acc_adam_algorithm_hist

                self.hj_mad.beta = 0.0
            acc_string = ' (Accelerated)'
        else:
            acc_string = ' (Non-Accelerated)'
        
        _, tk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist = algorithm_hist

        plt.figure(figsize=(10, 8))

        # Error history subplot
        plt.subplot(2, 2, 1)
        plt.semilogy(xk_error_hist, label='Error History'+acc_string, color='blue')
        if non_acc_algorithm_hist is not None:
            plt.semilogy(non_acc_xk_error_hist, label='Error History (Non-Accelerated)', color='orange', linestyle='--')
            if not self.hj_mad.trap_integration:
                plt.semilogy(adam_xk_error_hist, label='Error History (Non-Accelerated-Adam)', color='red', linestyle=':')
                plt.semilogy(acc_adam_xk_error_hist, label='Error History (Accelerated-Adam)', color='green', linestyle='-.')
        plt.title('Error History')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.legend(loc='upper right')

        # f(k) history subplot
        plt.subplot(2, 2, 2)
        plt.plot(fk_hist, label='f(k) History'+acc_string, color='blue')
        if non_acc_algorithm_hist is not None:
            plt.plot(non_acc_fk_hist, label='f(k) History (Non-Accelerated)', color='orange', linestyle='--')

            if not self.hj_mad.trap_integration:
                plt.plot(adam_fk_hist, label='f(k) History (Non-Accelerated-Adam)', color='red', linestyle=':')
                plt.plot(acc_adam_fk_hist, label='f(k) History (Accelerated-Adam)', color='green', linestyle='-.')
        plt.title('f(k) History')
        plt.xlabel('Iterations')
        plt.ylabel('f(k)')
        plt.legend(loc='upper right')

        # t(k) history subplot
        plt.subplot(2, 2, 3)
        plt.semilogy(tk_hist, label='t(k) History'+acc_string, color='blue')
        if non_acc_algorithm_hist is not None:
            plt.semilogy(non_acc_tk_hist, label='t(k) History (Non-Accelerated)', color='orange', linestyle='--')

            if not self.hj_mad.trap_integration:
                plt.semilogy(adam_tk_hist, label='t(k) History (Non-Accelerated-Adam)', color='red', linestyle=':')
                plt.semilogy(acc_adam_tk_hist, label='t(k) History (Accelerated-Adam)', color='green', linestyle='-.')
        plt.title('t(k) History')
        plt.xlabel('Iterations')
        plt.ylabel('t(k)')
        plt.legend(loc='best')

        # Relative gradient norm history subplot
        plt.subplot(2, 2, 4)
        plt.semilogy(rel_grad_uk_norm_hist, label='Rel. Grad Norm'+acc_string, color='blue')
        if non_acc_algorithm_hist is not None:
            plt.semilogy(non_acc_rel_grad_uk_norm_hist, label='Rel. Grad Norm (Non-Accelerated)', color='orange', linestyle='--')

            if not self.hj_mad.trap_integration:
                plt.semilogy(adam_rel_grad_uk_norm_hist, label='Rel. Grad Norm (Non-Accelerated-Adam)', color='red', linestyle=':')
                plt.semilogy(acc_adam_rel_grad_uk_norm_hist, label='Rel. Grad Norm (Accelerated-Adam)', color='green', linestyle='-.')
        plt.title('Relative Gradient Norm History')
        plt.xlabel('Iterations')
        plt.ylabel('Relative Gradient Norm')
        plt.legend(loc='best')

        plt.tight_layout()  # Adjust the layout
        plt.savefig("MAD_combined_history_plot.png", format='png', bbox_inches='tight')
        plt.show()
        plt.close()  # Free up memory

        # Display the latest saved plot
        with self.plot_results_output:
            clear_output(wait=True)  # Clear previous plot
            display(Image(filename="MAD_combined_history_plot.png"))

    def reset_plot(self):
        """Reset the plot and hide the plot results."""
        self.plot_hj_mad()
        self.plot_results_output.clear_output()  # Clear previous results
        #self.plot_results_output = None  # Reset the plot results
    
    def tol_input_update(self, _):
        self.hj_mad.tol = self.tol_input.value

    def print_slider_values(self):
        """Print the values of the sliders and selected function in a rotated table format."""
        function_name = self.function_dropdown.value
        slider_values = {
            'x_0': self.x_0_slider.value,
            't_init': self.t_init_slider.value,
            't_max': self.t_max_slider.value,
            'delta': self.delta_slider.value,
            'int_samples': self.int_samples_slider.value,
            'max_iters': self.max_iters_slider.value,
            'alpha': self.alpha_slider.value,
            't_threshold': self.t_threshold
        }

        # Create a smaller HTML table in rotated format
        html_content = "<h3>Slider Values and Function Name</h3><table style='border-collapse: collapse; width: 50%; font-size: 14px;'><tr><th style='border: 1px solid black;'><strong>Parameter</strong></th><th style='border: 1px solid black;'><strong>Value</strong></th></tr>"
        
        # Add each parameter and its value as a new row
        for param, value in slider_values.items():
            html_content += f"<tr><td style='border: 1px solid black;'>{param}</td><td style='border: 1px solid black;'>{value:.2f}</td></tr>"
        
        # Add the function name at the end
        html_content += f"<tr><td style='border: 1px solid black;'>Function</td><td style='border: 1px solid black;'>{function_name}</td></tr>"
        html_content += "</table>"

        # Display the table
        display(HTML(html_content))



