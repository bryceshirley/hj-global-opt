import torch

def isotropic_one_norm(tensor):
    # Compute Euclidean norm along the channel dimension
    euclidean_norm = torch.norm(tensor, p=2, dim=1)
    # Compute one norm across the remaining dimensions
    one_norm = torch.sum(euclidean_norm)
    return one_norm