import matplotlib.pyplot as plt
from numpy import float32, load
from operators import Finite_Difference_Gradient_2D
from optimisation import PDHG2
from proximal_maps import squared_l2_prox, l2ball_projection
from skimage import data
from skimage.color import rgb2gray
import torch
from hj_prox import HJ_prox

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if \
    torch.backends.mps.is_available() else 'cpu'
# device = 'cpu' # Uncomment this line to run on CPU

# Initialse data
torch.manual_seed(13)
# Load the astronaut image
astronaut_image = data.astronaut()
# Convert the image to a float32 numpy array and normalize to [0, 1]
astronaut_image = astronaut_image.astype(float32) / 255.0
# Convert the numpy array to a PyTorch tensor
astronaut_tensor = torch.from_numpy(astronaut_image)
# Add a batch dimension and permute channel dimension
astronaut_tensor = astronaut_tensor.unsqueeze(0).permute(0, 3, 1, 2)
# Add noise to the image
noise_level = 1e-1
data_tensor = astronaut_tensor + noise_level * torch.randn_like(astronaut_tensor, device='cpu')
# Move the tensor to the desired device (e.g., 'cpu', 'cuda' or 'mps')
data_tensor = data_tensor.to(device)

# Initialise regularisation parameter and operator
reg_param = 0.13
G = Finite_Difference_Gradient_2D()

# Initialise PDHG proximal maps and step sizes
step_sizes = [1/8, 1]
#data_fidelity = lambda argument, threshold: squared_l2_prox(argument, threshold/reg_param, data_tensor)
data_fidelity = lambda argument, threshold: HJ_prox(argument, t=threshold/reg_param)

regularisation_term = lambda argument, threshold: l2ball_projection(argument, axis=1)


# Initialise arguments
initial_arguments = [torch.zeros_like(data_tensor, dtype=data_tensor.dtype, device=device), torch.zeros_like(G(data_tensor), dtype=data_tensor.dtype, device=device)]

# Initialise PDHG
rof_pdhg = PDHG2([data_fidelity, regularisation_term], G, step_sizes, initial_arguments, max_iter=3000, tol=1e-6, print_output=100)

# Run PDHG
result = rof_pdhg()

# Visualise data and results in a single figure
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(data_tensor[0].permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.title("Noisy Image")
plt.subplot(1, 2, 2)
plt.imshow(result[2][0].permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.title("Reconstructed Image")
plt.show()