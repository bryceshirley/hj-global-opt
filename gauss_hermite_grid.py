import numpy as np
from itertools import product
from scipy.special import roots_hermite
import matplotlib.pyplot as plt

import numpy as np

def generate_gh_grid_matrix_with_threshold(n, m, sample_fraction=1):
    """
    Generates a sampled column matrix for n-dimensional Gauss-Hermite quadrature
    and removes rows with weights below a threshold.

    Args:
        n (int): Number of dimensions.
        m (int): Number of Gauss-Hermite nodes per dimension.
        sample_fraction (float): Fraction of samples to retain.

    Returns:
        np.ndarray: A matrix of size (filtered_samples, dimensions + 1), where
                    each row is [prod(w_i), z_i1, z_i2, ..., z_in].
    """
    # Generate 1D Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(m)

    # Calculate threshold θm
    w_min = weights[0]
    mid_idx = m // 2 if m % 2 == 1 else m // 2 - 1
    w_mid =weights[mid_idx]
    # print(f"{w_min=}, {w_mid=}")

    # # Compute theta_m
    theta_m = (w_min * w_mid) / m

    # Adjust weights to avoid zero
    #theta_m = 1e10  # Small offset to prevent zero weights

    # Compute logarithmic threshold
    #theta_m = np.exp(np.log(weights.min()) + np.log(weights.mean())) / m

    # Determine the total number of samples
    total_samples = int(sample_fraction * m**n)

    # Randomly sample points with uniform spread
    sampled_indices = np.random.choice(m, size=(total_samples, n))

    # Extract the sampled grid of nodes and weights
    z_grid = nodes[sampled_indices]  # Shape: (total_samples, n)
    w_grid = weights[sampled_indices]  # Shape: (total_samples, n)
   
   # Compute product of weights and filter by threshold
    w_prod = np.prod(w_grid, axis=1)
    print(w_prod.shape)
    print(np.min(w_prod))
    print(np.max(w_prod))
    # Print the largest 10 weights
    largest_weights = np.sort(w_prod.flatten())[-10:]  # Flatten, sort, and take the last 10
    print("The 10 largest weights are:", largest_weights)
    
    valid_mask = w_prod > -15
    
    z_grid = z_grid[valid_mask]
    w_prod = w_prod[valid_mask]

    print(np.min(w_prod))
    print(w_prod[int(w_prod.shape[0] / 2)])

    # Combine weights and transformed nodes
    grid_matrix = np.column_stack((w_prod, z_grid))

    return grid_matrix, theta_m

# Test for 2 dimensions
n = 2  # Number of dimensions
m = 1000  # Number of nodes per dimension
#rho = 0.0  # Correlation coefficient (-1,1) with 0 being no correlation
samples_fraction = 0.01
# Generate the grid matrix with threshold
grid_matrix, theta_m = generate_gh_grid_matrix_with_threshold(n, m,samples_fraction)

# Display the threshold and the resulting matrix
print("Threshold θm:", theta_m)
print("Percentage of Retained Points: {:.3e}%".format((grid_matrix.shape[0] / m**n) * 100))
print("Filtered Grid Matrix Shape:", grid_matrix.shape)
print("Number of samples with weights above threshold:", grid_matrix.shape[0])
print("")

if n == 2:
        # Extract z1 and z2 (node positions)
    z1 = grid_matrix[:, 1]
    z2 = grid_matrix[:, 2]

    # Plot the points
    plt.figure(figsize=(8, 8))
    plt.scatter(z1, z2, c='blue', s=10, label='Filtered Points')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title(f"Filtered Gauss-Hermite Points (2D, m={m})")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.legend()
    plt.grid(True)
    plt.show()


# def generate_gh_grid_matrix_with_threshold(n, m, rho=0):
#     """
#     Generates a column matrix for n-dimensional Gauss-Hermite quadrature
#     and removes rows with weights below a threshold.

#     Args:
#         n (int): Number of dimensions.
#         m (int): Number of Gauss-Hermite nodes per dimension.

#     Returns:
#         np.ndarray: A matrix of size (filtered_samples, dimensions + 1), where
#                     each row is [prod(w_i), z_i1, z_i2, ..., z_in].
#     """
#     # Generate 1D Gauss-Hermite nodes and weights
#     nodes, weights = np.polynomial.hermite.hermgauss(m)

#     # Calculate threshold θm
#     w_1 = weights[0]
#     w_mid = weights[m // 2] if m % 2 == 1 else weights[m // 2 - 1]
#     theta_m = w_1 * w_mid /m

#     # Create correlation matrix (all off-diagonal elements set to rho)
#     correlation_matrix = np.full((n, n), rho)
#     np.fill_diagonal(correlation_matrix, 1)

#     # Perform Cholesky decomposition
#     L = np.linalg.cholesky(correlation_matrix)

#     # Cartesian product for n-dimensional indices
#     indices = product(range(m), repeat=n)

#     # Initialize a list to store the grid
#     grid = []

#     for idx in indices:
#         print(f"{idx=}")
#         # Extract nodes and weights for the current indices
#         z = np.array([nodes[i] for i in idx])  # Corresponding z values
#         w = [weights[i] for i in idx]  # Corresponding weights
#         w_prod = np.prod(w)  # Product of weights              # Product of weights

#         # Apply the threshold
#         if w_prod <= theta_m:
#             continue
#             # Append [prod(w), z_1, z_2, ..., z_n] as a row
#             #grid.append([w_prod] + z)

#         # Apply the correlation transformation using the Cholesky decomposition
#         correlated_z = np.dot(L, z)  # Apply the linear transformation

#         # Append [prod(w), z_1, z_2, ..., z_n] as a row
#         grid.append([w_prod] + correlated_z.tolist())

#     # Convert to a numpy array
#     grid_matrix = np.array(grid)

#     # Randomly remove 80% of the samples
#     num_samples = grid_matrix.shape[0]
#     num_samples_to_remove = int(0.99 * num_samples)

#     # Get random indices to keep 20% of the samples
#     keep_indices = np.random.choice(num_samples, num_samples - num_samples_to_remove, replace=False)

#     # Filter the weight products and grid points using the selected indices
#     grid_matrix = grid_matrix[keep_indices]

#     return grid_matrix, theta_m

# def generate_smolyak_sparse_grid(n, m, rho, level, prune=True):
#     """
#     Generates a Smolyak sparse grid with correlation incorporated, using Gauss-Hermite quadrature.

#     Args:
#         n (int): Number of dimensions.
#         m (int): Number of Gauss-Hermite nodes per dimension.
#         rho (float): The correlation coefficient between dimensions (between -1 and 1).
#         level (int): The level of the Smolyak sparse grid.
#         prune (bool): Whether to prune the grid based on the weight threshold θ_m.

#     Returns:
#         np.ndarray: A matrix of size (filtered_samples, dimensions + 1), where each row
#                     is [prod(w_i), z_i1, z_i2, ..., z_in].
#     """
#     # Generate 1D Gauss-Hermite nodes and weights
#     nodes, weights = np.polynomial.hermite.hermgauss(m)

#     # Calculate the threshold θ_m
#     theta_m = (weights[0] * weights[(m + 1) // 2 - 1]) / m

#     # Create correlation matrix (all off-diagonal elements set to rho)
#     correlation_matrix = np.full((n, n), rho)
#     np.fill_diagonal(correlation_matrix, 1)

#     # Perform Cholesky decomposition
#     L = np.linalg.cholesky(correlation_matrix)

#     # Smolyak grid construction: sum grids with different levels of resolution
#     # Start with an empty grid list
#     grid = []

#     # Loop over different resolution levels
#     for l in range(1, level + 1):
#         # Cartesian product for the 1D Gauss-Hermite nodes with resolution level l
#         indices = product(range(m), repeat=n)

#         for idx in indices:
#             # Extract nodes and weights for the current indices
#             z = np.array([nodes[i] for i in idx])  # Corresponding z values
#             w = [weights[i] for i in idx]  # Corresponding weights
#             w_prod = np.prod(w)  # Product of weights

#             # Only include the point if the product of weights exceeds the threshold (if prune=True)
#             if prune and w_prod < theta_m:
#                 continue

#             # Apply the correlation transformation using the Cholesky decomposition
#             correlated_z = np.dot(L, z)  # Apply the linear transformation

#             # Append [prod(w), z_1, z_2, ..., z_n] as a row
#             grid.append([w_prod] + correlated_z.tolist())

#     # Convert to a numpy array and remove duplicate points by converting to a set
#     grid_matrix = np.unique(np.array(grid), axis=0)

#     return grid_matrix


# # Specify dimensions, nodes, correlation, and Smolyak grid level
# n = 4  # Number of dimensions
# m = 10  # Number of nodes per dimension
# rho = 0.5  # Desired correlation coefficient
# level = 8  # Smolyak grid level

# # Generate the Smolyak sparse grid with correlation
# grid_matrix = generate_smolyak_sparse_grid(n, m, rho, level)

# # Display the shape of the matrix and a few rows
# print("Smolyak Sparse Grid Matrix Shape:", grid_matrix.shape)
# print("Number of samples with weights above threshold:", grid_matrix.shape[0])

# if n ==2:
#     # Plot the Smolyak sparse grid in 2D
#     z1 = grid_matrix[:, 1]
#     z2 = grid_matrix[:, 2]
#     plt.figure(figsize=(8, 6))
#     plt.scatter(z1, z2, color='blue', marker='o', edgecolor='black', alpha=0.7)
#     plt.title('Smolyak Sparse Grid in 2D', fontsize=16)
#     plt.xlabel('X-axis', fontsize=12)
#     plt.ylabel('Y-axis', fontsize=12)
#     plt.grid(True)
#     plt.show()