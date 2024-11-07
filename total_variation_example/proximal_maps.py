import torch

def l2ball_projection(argument, axis=None):
    """
    Projects the input tensor onto the L2 ball.

    Parameters:
    argument (torch.Tensor): The input tensor to be projected.
    axis (int, optional): The axis along which to compute the L2 norm. If None, the projection is applied element-wise.

    Returns:
    torch.Tensor: The projected tensor, where each element or sub-tensor (depending on the axis) is within the L2 ball.
    """
    if axis is None:
        return torch.clamp(argument, min=-1.0, max=1.0)
    else:
        denominator = torch.clamp(torch.norm(argument, p=2, dim=axis), min=1.0)
        return argument / torch.unsqueeze(denominator, dim=axis)
    
def l2_norm_prox(argument, threshold):
    """
    Compute the proximal operator for the L2 norm.

    This function computes the proximal operator for the L2 norm, which is used 
    in optimisation problems involving regularisation terms. The proximal operator 
    is defined as:

        prox_{threshold * ||.||_2}(argument) = argument * (1 - threshold / max(||argument||_2, threshold))

    Args:
        argument (torch.Tensor): The input tensor for which the proximal operator is computed.
        threshold (float): The threshold value used in the proximal operator.

    Returns:
        torch.Tensor: The result of applying the proximal operator to the input tensor.
    """
    norm_argument = torch.norm(argument, p=2, dim=-1, keepdim=True)
    norm_argument = torch.maximum(norm_argument, torch.tensor(threshold))
    return argument * (1 - threshold / norm_argument)

def soft_thresholding(argument, threshold):
    """
    Applies the soft thresholding operation to the input tensor.

    The soft thresholding function is defined as:
        soft_thresholding(x, t) = sign(x) * max(|x| - t, 0)

    Args:
        argument (torch.Tensor): The input tensor to be thresholded.
        threshold (float): The threshold value.

    Returns:
        torch.Tensor: The result of applying the soft thresholding operation to the input tensor.
    """
    abs_argument = torch.abs(argument)
    sign_argument = torch.sign(argument)
    output = sign_argument * torch.clamp(abs_argument - threshold, min=0)
    return output

def squared_l2_prox(argument, threshold, data):
    """
    Compute the proximal operator for the squared L2 norm.

    The proximal operator for the squared L2 norm is given by:
    prox_{threshold * ||.||^2}(argument) = (argument + threshold * data) / (1 + threshold)

    Args:
        argument (torch.Tensor): The input value or array for which the proximal operator is computed.
        threshold (float): The regularization parameter.
        data (torch.Tensor): The data term to be incorporated in the proximal operator.

    Returns:
        torch.Tensor: The result of the proximal operator computation.
    """
    return (argument + threshold * data) / (1 + threshold)

def subsampled_orthogonal_prox(argument, mask, OM, data, threshold):    
    return OM.T @ ((OM @ argument + threshold * data) / (1 + threshold * mask))