import numpy as np
import torch


def Sin_numpy(x):
    """Compute the sine of x using numpy."""
    # Global minima at x=3*pi/2 + 2*k*pi where k is an integer, gamma = None
    return np.sin(x)

def Sinc_numpy(x):
    """Compute the sinc function: sin(x)/(x) using numpy."""
    # Global minima at x=4.49341 and -4.49341, gamma = 2.75909 (restrict x0>0 for positive min)
    return np.sinc(x/np.pi)

def MultiMinimaFunc_numpy(x, noise_level=0.05):
    """Compute a multi-minima function with added noise: 5*sin(x) + 0.1*x^2 + noise.
    
    Args:
        x (float or np.ndarray): Input value(s) for the function.
        noise_level (float): The standard deviation of the noise to add. Default is 0.05.
    
    Returns:
        float or np.ndarray: The value of the function with added noise.
    """
    # Global minima at x=--1.51034569, gamma = 1.89777
    base_function = 5 * np.sin(x) + 0.1 * x**2
    noise = 0 #np.random.normal(0, noise_level, size=x.shape if isinstance(x, np.ndarray) else 1)
    return base_function + noise

def MultiMinimaAbsFunc_numpy(x):
    """Compute a non-differentiable multi-minima function: 5*|sin(x)| + 0.1*x^2."""
    # Global minima at (1,0.1), gamma = 1.43457
    return 5 * np.abs(np.sin(0.5*x-1)) + 0.1 * x**2

def DiscontinuousFunc_numpy(x):
    """Compute a discontinuous function with a global minimum."""
    # Global minima at x=2, gamma = 7
    return np.where(x < 2, 5 * np.sin(x) + 0.1 * x**2,  0.5*(x - 2)**2-20)#np.where(x < 2, np.abs(x**2/(x+1) + 80)+x**2,  0.5*(x - 2)**2+2)

