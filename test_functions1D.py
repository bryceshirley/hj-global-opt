import torch
import numpy as np

# 1D functions
# Sine function
def Sin(x):
    return torch.sin(x)

def Sinc(x):
    return torch.where(x == 0, torch.tensor(1.0), torch.sin(x) / x)

# Example of a 1D function with multiple local minima and one global minimum
def MultiMinimaFunc(x):
    return 5*torch.sin(x) + 0.1*x**2

# ----------------- Numpy Versions -----------------
def Sin_numpy(x):
    return np.sin(x)

def Sinc_numpy(x):
    return np.sinc(x / np.pi)  # np.sinc is normalized as sin(pi*x)/(pi*x)

def MultiMinimaFunc_numpy(x):
    return 5*np.sin(x) + 0.1*x**2

