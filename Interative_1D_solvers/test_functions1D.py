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

def MultiMinimaFunc_numpy(x):
    """Compute a multi-minima function with added noise: 5*sin(x) + 0.1*x^2 + noise.
    
    Args:
        x (float or np.ndarray): Input value(s) for the function.
    
    Returns:
        float or np.ndarray: The value of the function with added noise.
    """
    # Global minima at x=--1.51034569, gamma = 1.89777
    return 5 * np.sin(x+10) + 0.1 * (x+10)**2

def MultiMinimaAbsFunc_numpy(x):
    """Compute a non-differentiable multi-minima function: 5*|sin(x)| + 0.1*x^2."""
    # Global minima at (1,0.1), gamma = 1.43457
    return 5 * np.abs(np.sin(0.5*x-1)) + 0.1 * x**2

def DiscontinuousFunc_numpy(x):
    """Compute a discontinuous function with a global minimum."""
    # Global minima at x=2, gamma = 7
    return np.where(x < 2, 5 * np.sin(x) + 0.1 * x**2,  0.5*(x - 2)**2-20)#np.where(x < 2, np.abs(x**2/(x+1) + 80)+x**2,  0.5*(x - 2)**2+2)

def Drop_Wave_1D(x):
    """Compute the Drop Wave function in 1D."""
    return -(1 + np.cos(12 * np.sqrt(x**2))) / (0.5 * x**2 + 2)