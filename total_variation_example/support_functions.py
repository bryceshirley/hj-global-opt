import torch
from operators import Finite_Difference_Gradient_2D

def calculate_norm_difference(arguments, prev_arguments):
    if len(arguments) != len(prev_arguments):
        raise ValueError("arguments and prev_arguments must have the same length")
    norm_diff = sum(torch.norm(arg - prev_arg)/torch.norm(arg) if torch.norm(arg) != 0 \
                        else torch.norm(arg - prev_arg) for arg, prev_arg in zip(arguments, prev_arguments))
    norm_diff /= len(arguments)
    return norm_diff

def isotropic_one_norm(tensor):
    # Compute Euclidean norm along the channel dimension
    euclidean_norm = torch.norm(tensor, p=2, dim=1)
    # Compute one norm across the remaining dimensions
    one_norm = torch.sum(euclidean_norm)
    return one_norm

def isotropic_total_variation(tensor):
    return isotropic_one_norm(Finite_Difference_Gradient_2D(tensor))