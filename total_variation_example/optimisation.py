from support_functions import calculate_norm_difference, isotropic_total_variation

class Alternating_Optimisation():
    """
    A class to perform alternating optimization using a list of optimizers.
    Attributes:
    -----------
    optimisers : list
        A list of optimiser functions to be applied in sequence.
    arguments : list
        Initial arguments for the optimisers.
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    tol : float, optional
        Tolerance for the stopping criterion (default is 1e-6).
    print_output : int, optional
        Frequency of printing the output (default is None, meaning no output).
    Methods:
    --------
    __call__():
        Executes the alternating optimisation process.
    restart_counter():
        Resets the iteration counter to zero.
    sensitivity():
        Computes the sensitivity value. This method should be implemented by subclasses.
    stopping_criterion(sensitivity_value):
        Checks if the stopping criterion is met based on the sensitivity value.
    """

    def __init__(self, optimisers, initial_arguments, max_iter=1000, tol=1e-6, print_output=None):
        self.optimisers = optimisers
        self.arguments = initial_arguments.copy()
        self.max_iter = max_iter
        self.tol = tol
        self.print_output = print_output
        self.counter = 0

    def __call__(self):
        sensitivity_value = float("inf")
        while self.counter < self.max_iter and not self.stopping_criterion(sensitivity_value):
            for j, optimiser in enumerate(self.optimisers):
                self.arguments[j] = optimiser(self.arguments)
            sensitivity_value = self.sensitivity()
            if self.print_output is not None and (1 + self.counter) % self.print_output == 0:
                print(f"Iteration: {self.counter + 1}, Sensitivity: {sensitivity_value}")
            self.counter += 1
        if self.print_output is not None:
            print(f"Total Iterations: {self.counter}, Sensitivity: {sensitivity_value}/{self.tol}")
        return self.arguments
    
    def restart_counter(self):
        self.counter = 0

    def sensitivity(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def stopping_criterion(self, sensitivity_value):
        return sensitivity_value < self.tol
    
class PDHG(Alternating_Optimisation):
    """
    Primal-Dual Hybrid Gradient (PDHG) optimisation algorithm.

    This class implements the PDHG optimisation algorithm, which is a method for solving convex-concave saddle-point problems. It alternates between updating primal and dual variables to find an optimal solution.

    Attributes:
        updates (tuple): A tuple containing the primal and dual update functions.
        initial_arguments (list): A list of initial arguments for the optimisation.
        max_iter (int): Maximum number of iterations for the optimisation. Default is 1000.
        tol (float): Tolerance for convergence. Default is 1e-6.
        print_output (callable, optional): A function to print output during optimisation. Default is None.

    Methods:
        sensitivity():
            Computes the sensitivity of the current solution. This is defined as the norm difference between the current and previous primal variables.
    """

    def __init__(self, updates, initial_arguments, max_iter=1000, tol=1e-6, print_output=None):
        primal_update, dual_update = updates
        prev_primal_update = lambda argument: argument[2].clone()
        prev_dual_update = lambda argument: argument[3].clone()
        optimisers = [prev_primal_update, prev_dual_update, primal_update, dual_update]
        extended_initial_arguments = [initial_arguments[0].clone(), initial_arguments[1].clone(), initial_arguments[0].clone(), initial_arguments[1].clone()]
        super().__init__(optimisers, extended_initial_arguments, max_iter, tol, print_output)

    def sensitivity(self):
        if self.counter == 0:
            norm_diff = float("inf")
        else:
            norm_diff = calculate_norm_difference(self.arguments[0:1], self.arguments[2:3])
        return norm_diff
    
class PDHG2(PDHG):

    def __init__(self, proximal_maps, linear_operator, step_sizes, initial_arguments, max_iter=1000, tol=1e-6, print_output=None):
        primal_update = lambda argument: proximal_maps[0](argument[2] - step_sizes[0] * (linear_operator.T @ argument[3]), step_sizes[0])
        dual_update = lambda argument: proximal_maps[1](argument[3] + step_sizes[1] * (linear_operator @ (2*argument[2] - argument[0])), step_sizes[1])
        super().__init__([primal_update, dual_update], initial_arguments, max_iter, tol, print_output)