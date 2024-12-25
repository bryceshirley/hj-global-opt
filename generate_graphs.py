import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Style settings for a polished look
mpl.style.use('seaborn-darkgrid')
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# Define the regularization term
def regularization_term(x, z,t):
    return 1/(2*t) * (x - z) ** 2

# Define the Moreau envelope
def moreau_envelope(f, x, t, z):
    return np.min(f(z) + regularization_term(x, z, t))

## CONVEX BUT NON-SMOOTH FUNCTION

# Define the non-smooth function f(z)
def f(z):
    return np.abs(z)

# Generate z values
z = np.linspace(-6, 6, 400)

# Fixed point x to visualize around
x_fixed = 2
t=100
y_f = f(z)  # Non-smooth function values
y_reg = regularization_term(x_fixed, z,t)  # Regularization term values
y_combined = y_f + y_reg  # Combined function values

# Create the plot
plt.figure(figsize=(12, 7))

# Plot non-smooth function f(z)
plt.plot(z, y_f, label='Non-Smooth Function $f(z) = |z|$', 
         color='red', alpha=0.8)

# Plot combined function
plt.plot(z, y_combined, label='Combined Function $f(z) + \\frac{1}{2}(x - z)^2$', 
         color='blue', alpha=0.8)

# Title and labels
#plt.title('Proximal Point Minimization Visualization', fontsize=16)
# plt.xlabel('z', fontsize=14)
# plt.ylabel('Value', fontsize=14)

# Mark the fixed point x
plt.scatter(x_fixed, f(x_fixed), color='black', marker='x', label='Fixed $x$', zorder=5, s=100)

# Indicate the minimum point of the combined function
min_index = np.argmin(y_combined)
min_z = z[min_index]
plt.scatter(min_z, y_combined[min_index], color='green', s=100, marker='x', label='Minimum Point', zorder=5)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add x and y axis lines
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# Add text labels next to the curves
plt.text(-4, f(-3.9), "$\\mathbf{f(x) = |x|}$", color='red', fontsize=16, verticalalignment='top', horizontalalignment='right')
plt.text(-2.5, 4, "$\\mathbf{f(x) + \\frac{1}{2t}||x - x_0||^2}$", color='blue', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(2.2, 2, "$\\mathbf{x_0}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(0.25, 1.95, "Minimum Point", color='green', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(5, -0.1, "$\\mathbf{x}$", color='black', fontsize=16, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.1, 5.3, "$\\mathbf{y}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')

# Adjust spines to place numbers on the axes
ax = plt.gca()  # Get current axis
ax.spines['left'].set_position('zero')  # Set y-axis spine to zero
ax.spines['bottom'].set_position('zero')  # Set x-axis spine to zero
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

# Grid and layout
plt.grid()
plt.tight_layout()
plt.ylim(-0.5, 5)
plt.xlim(-5,5)

# Show plot
plt.show()

# Compute the Moreau envelope at each point
y_moreau = np.array([moreau_envelope(f, x_val, t,z) for x_val in z])

# Create the plot
plt.figure(figsize=(12, 7))

# Plot non-smooth function f(z)
plt.plot(z, y_f, label=r'Non-Smooth Function $f(z) = 5\sin(z) + 0.1z^2 + 5$', 
         color='red', alpha=0.8)

# Plot combined function (proximal point minimization)
plt.plot(z, y_combined, label=r'Combined Function $f(z) + \frac{1}{2}(x - z)^2$', 
         color='blue', alpha=0.8)

# Plot the Moreau envelope
plt.plot(z, y_moreau, label=r'Moreau Envelope $e_t(f)(x)$', color='orange', linestyle='--', alpha=0.8)
plt.scatter(x_fixed, y_moreau[np.argmin(np.abs(z - x_fixed))], color='purple', s=100, marker='x', label='Moreau Point', zorder=5)

# Mark the fixed point x
plt.scatter(x_fixed, f(x_fixed), color='black', marker='x', label='Fixed $x$', zorder=5, s=100)

# Indicate the minimum point of the combined function
min_index = np.argmin(y_combined)
min_z = z[min_index]
plt.scatter(min_z, y_combined[min_index], color='green', s=100, marker='x', label='Minimum Point', zorder=5)

# Add dotted lines linking the proximal operator and the Moreau envelope
plt.plot([x_fixed, x_fixed], [f(x_fixed), y_moreau[np.argmin(np.abs(z - x_fixed))]], color='black', linestyle=':', label='Proximal Link', alpha=0.8)
plt.plot([min_z, x_fixed], [y_moreau[np.argmin(np.abs(z - x_fixed))], y_moreau[np.argmin(np.abs(z - x_fixed))]], color='black', linestyle=':', label='Proximal Link', alpha=0.8)


# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add x and y axis lines
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# Add text labels next to the curves
plt.text(-3, 4.5, "$\\mathbf{f(x) = |x|}$", color='red', fontsize=16, verticalalignment='top', horizontalalignment='right')
plt.text(-2.5, 4, "$\\mathbf{f(x) + \\frac{1}{2t}||x - x_0||^2}$", color='blue', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(2.2, 2, "$\\mathbf{x_0}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(min_z, 1.8, "$\\mathbf{\\hat{x}_0}$", color='green', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(5, -0.1, "$\\mathbf{x}$", color='black', fontsize=16, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.1, 5.3, "$\\mathbf{y}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(1, 0.5, "Moreau Envelope $\\mathbf{u(x,t)}$", color='orange', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(x_fixed, y_moreau[np.argmin(np.abs(z - x_fixed))], "$\\mathbf{u(x_0,t)}$", color='purple', fontsize=16, verticalalignment='top', horizontalalignment='left')


# Adjust spines to place numbers on the axes
ax = plt.gca()  # Get current axis
ax.spines['left'].set_position('zero')  # Set y-axis spine to zero
ax.spines['bottom'].set_position('zero')  # Set x-axis spine to zero
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

# Grid and layout
plt.grid()
plt.tight_layout()
plt.ylim(-0.5, 5)
plt.xlim(-5,5)

# Show plot
# plt.legend(fontsize=12)
plt.show()


## Non-CONVEX BUT NON-SMOOTH FUNCTION

# Define the non-smooth function f(z)
def f(z):
    return 5*np.abs(np.sin(z)) + 0.1*z**2 +5

# Generate z values
z = np.linspace(-22, 22, 400)

# Fixed point x to visualize around
x_fixed = 15
t=5
y_f = f(z)  # Non-smooth function values
y_reg = regularization_term(x_fixed, z,t)  # Regularization term values
y_combined = y_f + y_reg  # Combined function values

# Create the plot
plt.figure(figsize=(12, 7))

# Plot non-smooth function f(z)
plt.plot(z, y_f, label='Non-Smooth Function $f(z) = |z|$', 
         color='red', alpha=0.8)

# Plot combined function
plt.plot(z, y_combined, label='Combined Function $f(z) + \\frac{1}{2}(x - z)^2$', 
         color='blue', alpha=0.8)

# Title and labels
#plt.title('Proximal Point Minimization Visualization', fontsize=16)
# plt.xlabel('z', fontsize=14)
# plt.ylabel('Value', fontsize=14)

# Mark the fixed point x
plt.scatter(x_fixed, f(x_fixed), color='black', marker='x', label='Fixed $x$', zorder=5, s=100)

# Indicate the minimum point of the combined function
min_index = np.argmin(y_combined)
min_z = z[min_index]
plt.scatter(min_z, y_combined[min_index], color='green', s=100, marker='x', label='Minimum Point', zorder=5)

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add x and y axis lines
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# Add text labels next to the curves
# plt.text(-15, f(-15), "$\\mathbf{f(x)}$", color='red', fontsize=16, verticalalignment='top', horizontalalignment='right')
# plt.text(2.5, f(10)+regularization_term(x_fixed,10,t), "$\\mathbf{f(x) + \\frac{1}{2t}||x - x_0||^2}$", color='blue', fontsize=16, verticalalignment='top', horizontalalignment='left')
# plt.text(x_fixed, f(x_fixed)-1, "$\\mathbf{x_0}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
# plt.text(min_z-5, y_combined[min_index]-1, "Minimum Point", color='green', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(20, -0.1, "$\\mathbf{x}$", color='black', fontsize=16, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.1, 41, "$\\mathbf{y}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')


# Adjust spines to place numbers on the axes
ax = plt.gca()  # Get current axis
ax.spines['left'].set_position('zero')  # Set y-axis spine to zero
ax.spines['bottom'].set_position('zero')  # Set x-axis spine to zero
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

# Grid and layout
plt.grid()
plt.tight_layout()
plt.ylim(-0.5, 40)
plt.xlim(-20,20)

# Show plot
plt.show()

# Generate z values
z = np.linspace(-22, 22, 400)

# Fixed point x to visualize around
x_fixed = 15
t = 5
y_f = f(z)  # Non-smooth function values
y_reg = regularization_term(x_fixed, z, t)  # Regularization term values
y_combined = y_f + y_reg  # Combined function values

# Compute the Moreau envelope at each point
y_moreau = np.array([moreau_envelope(f, x_val, t,z) for x_val in z])

# Create the plot
plt.figure(figsize=(12, 7))

# Plot non-smooth function f(z)
plt.plot(z, y_f, label=r'Non-Smooth Function $f(z) = 5\sin(z) + 0.1z^2 + 5$', 
         color='red', alpha=0.8)

# Plot combined function (proximal point minimization)
plt.plot(z, y_combined, label=r'Combined Function $f(z) + \frac{1}{2}(x - z)^2$', 
         color='blue', alpha=0.8)

# Plot the Moreau envelope
plt.plot(z, y_moreau, label=r'Moreau Envelope $e_t(f)(x)$', color='orange', linestyle='--', alpha=0.8)
plt.scatter(x_fixed, y_moreau[np.argmin(np.abs(z - x_fixed))], color='purple', s=100, marker='x', label='Moreau Point', zorder=5)

# Mark the fixed point x
plt.scatter(x_fixed, f(x_fixed), color='black', marker='x', label='Fixed $x$', zorder=5, s=100)

# Indicate the minimum point of the combined function
min_index = np.argmin(y_combined)
min_z = z[min_index]
plt.scatter(min_z, y_combined[min_index], color='green', s=100, marker='x', label='Minimum Point', zorder=5)

# Add dotted lines linking the proximal operator and the Moreau envelope
plt.plot([x_fixed, x_fixed], [f(x_fixed), y_moreau[np.argmin(np.abs(z - x_fixed))]], color='black', linestyle=':', label='Proximal Link', alpha=0.8)
plt.plot([min_z, x_fixed], [y_moreau[np.argmin(np.abs(z - x_fixed))], y_moreau[np.argmin(np.abs(z - x_fixed))]], color='black', linestyle=':', label='Proximal Link', alpha=0.8)

# Indicate Global Minimum
min_index_f = np.argmin(y_f)
min_z_f = z[min_index_f]
plt.scatter(min_z_f, y_f[min_index_f], color='dimgrey', s=100, marker='x', label='Minimum Point', zorder=5)
plt.text(-7, 4.5, "Global Minima $\\mathbf{x^{*}}$", color='dimgrey', fontsize=16, verticalalignment='top', horizontalalignment='left')

# Customize ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add x and y axis lines
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)

# Add text labels next to the curves
plt.text(-15, f(-15)-1, "$\\mathbf{f(x)}$", color='red', fontsize=16, verticalalignment='top', horizontalalignment='right')
plt.text(-9, 35, "$\\mathbf{f(x) + \\frac{1}{2t}||x - x_0||^2}$", color='blue', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(x_fixed, f(x_fixed)-1, "$\\mathbf{x_0}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(min_z-0.15, 15.5, "$\\hat{x}_0$", color='green', fontsize=16, verticalalignment='top', horizontalalignment='left')

plt.text(20, -0.1, "$\\mathbf{x}$", color='black', fontsize=16, verticalalignment='bottom', horizontalalignment='left')
plt.text(0.1, 41, "$\\mathbf{y}$", color='black', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(-20, 9, "Moreau Envelope $\\mathbf{u(x,t)}$", color='orange', fontsize=16, verticalalignment='top', horizontalalignment='left')
plt.text(x_fixed+1, y_moreau[np.argmin(np.abs(z - x_fixed))]+1, "$\\mathbf{u(x_0,t)}$", color='purple', fontsize=16, verticalalignment='top', horizontalalignment='left')

# Adjust spines to place numbers on the axes
ax = plt.gca()  # Get current axis
ax.spines['left'].set_position('zero')  # Set y-axis spine to zero
ax.spines['bottom'].set_position('zero')  # Set x-axis spine to zero
ax.spines['top'].set_visible(False)  # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine

# Grid and layout
plt.grid()
plt.tight_layout()
plt.ylim(-0.5, 40)
plt.xlim(-20, 20)

# Show plot
# plt.legend(fontsize=12)
plt.show()

