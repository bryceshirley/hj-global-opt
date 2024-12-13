import numpy as np
import matplotlib.pyplot as plt
from test_functions1D import (DiscontinuousFunc_numpy,
                              MultiMinimaAbsFunc_numpy, MultiMinimaFunc_numpy,
                              Sin_numpy, Sinc_numpy)
from scipy.optimize import differential_evolution

# Define the h function
def h(z, h_parameters):
    f, T, x = h_parameters  # Unpack the parameters (f = selected_function, T = fixed_T)
    return f(z) + (1 / (2 * T)) * (z - x)**2

# Define the objective function
def objective(delta, f, x, t, prox_ft, n_samples=100000):
    # Sample from N(x, delta*t)
    y_samples = np.random.normal(x, delta * t, n_samples)
    
    # Compute the two expectations
    exponent = -f(y_samples) / delta
    shifted_exponent = exponent - np.max(exponent)  # Numerical stability
    exp_values = np.exp(shifted_exponent)
    exp_values_y = y_samples * exp_values
    
    # Compute the ratio
    E1 = np.mean(exp_values_y)
    E2 = np.mean(exp_values)
    ratio = E1 / E2
    
    # Minimize the absolute difference from prox_ft(x)
    return np.linalg.norm(ratio - prox_ft)

# Parameters
x = 25.0  # Point of interest
t = 1  # Time step

# Optimize to find prox_ft
hk_parameters = (MultiMinimaFunc_numpy, t, x)
result = differential_evolution(h, [(-30, 30)], args=(hk_parameters,))
prox_ft = result.x[0]

# Grid search for delta
delta_values = np.linspace(0.001, 2.0, 300)  # Search range for delta
errors = [objective(delta, MultiMinimaFunc_numpy, x, t, prox_ft) for delta in delta_values]
best_delta = delta_values[np.argmin(errors)]

print("Optimal delta:", best_delta)

# Plot the objective function
plt.figure(figsize=(10, 6))
plt.semilogy(delta_values, errors, label="Objective Function")
#plt.semilogy(delta_values, (1/t)*(delta_values-best_delta)**2, label=r"(1/t)(δ+δ_{opt})^{2}")
plt.axvline(best_delta, color='r', linestyle='--', label=f"Optimal δ = {best_delta:.2f}")
plt.xlabel("δ (delta)")
plt.ylabel("Objective Function Value")
plt.title(f"Estimated Prox Objective Function vs. δ (Delta)\n x={x},t={t}")
plt.legend()
plt.grid()
plt.show()
