import numpy as np

def Sin_numpy(x):
    """Compute the sine of x using numpy."""
    # Global minima at x=3*pi/2 + 2*k*pi where k is an integer, gamma = pi 
    return np.sin(x)

def Sinc_numpy(x):
    """Compute the sinc function: sin(x)/(x) using numpy."""
    # Global minima at x=4.49341 and -4.49341, gamma = 2.75909 (restrict x0>0 for positive min)
    return np.sin(x)/x

def MultiMinimaFunc_numpy(x):
    """Compute a multi-minima function: 5*sin(x) + 0.1*x^2."""
    # Global minima at x=-1.51035, gamma = 3.14664
    return 5 * np.sin(x) + 0.1 * x**2

def MultiMinimaAbsFunc_numpy(x):
    """Compute a non-differentiable multi-minima function: 5*|sin(x)| + 0.1*x^2."""
    # Global minima at x=0, gamma = 1.63629
    return 5 * np.abs(np.sin(x)) + 0.1 * x**2
