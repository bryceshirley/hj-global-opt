import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from hj_mad_ls import HJ_MAD_ls

# Hyperparameters
n_neurons = 50  # Number of neurons in the shallow network
learning_rate = 0.01
num_epochs = 1000
batch_size = 100

# Generate data
no_of_samples = 500
noise_level = 1e-3
x = np.linspace(0, 2 * np.pi, no_of_samples)
y = np.sin(x) + noise_level * np.random.randn(*x.shape)  # Add noise to the samples

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the shallow neural network
class ShallowNet(nn.Module):
    def __init__(self, n_neurons):
        super(ShallowNet, self).__init__()
        self.linear = nn.Linear(1, n_neurons)
        self.coeffs = nn.Parameter(torch.randn(n_neurons, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)      # Compute w_j * x + b_j
        x = self.relu(x)        # Apply ReLU activation
        x = x @ self.coeffs     # Compute sum of c_j * ReLU(...)
        return x

# Initialize model, loss function, and optimizer
model = ShallowNet(n_neurons)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# HJ_MAD hyperparameters
delta = 10
t = 10e3
int_samples = 10e5
max_iters = 10e3
f_tol = 1e-5
sat_tol = 1e-5
delta_dampener=0.8
beta=0.5

# Use HJ_MAD to optimize the model parameters
HJ_MAD_alg = HJ_MAD_ls(delta, t, int_samples, max_iters, f_tol,sat_tol,delta_dampener,beta)

# Training loop: Iterate through the epochs
for epoch in range(num_epochs):
    # Iterate through the dataset in batches
    for inputs, targets in dataloader:
        # Define the function f (e.g., a loss function based on the model's predictions)
        def f(x):
            model_params = x  # x contains the model parameters
            model.set_parameters(model_params)  # Set the model parameters
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Calculate the loss (e.g., MSE)
            return loss.item()  # Return the loss value

        x_opt = HJ_MAD_alg.run(f, model.parameters())  # Run HJ_MAD to optimize the model parameters
        
        # Optionally, you can update the model with the optimized parameters if needed
        model.set_parameters(x_opt)  # Update model parameters with the optimized ones
        
    # Print the loss every 100 epochs for monitoring
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualise the learned function
plt.figure()
plt.plot(x, y, label='True function')
plt.plot(x, model(x_tensor).detach().numpy(), '--', label='Learned function')
plt.legend()
plt.show()