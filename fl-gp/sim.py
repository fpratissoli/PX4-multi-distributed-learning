import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
import time
import utilities as utils

warnings.filterwarnings("ignore")

# Define the true function
def f(x):
    return np.sin(x)

# Generate the data
np.random.seed(1)

# Training data
n = 500
X = np.random.uniform(0, 10, n)[:, np.newaxis]  # Changed range to 0-10
# Order the data
X = np.sort(X, axis=0)
y = f(X) + 0.1 * np.random.randn(n)[:, np.newaxis]

# Test data
X_ = np.linspace(0, 10, 100)[:, np.newaxis]
y_true = f(X_)
y_true = y_true.ravel()

# GP
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
tstart = time.time()
gp.fit(X, y)
print(f"Training time full GP: {time.time() - tstart:.2f}s")
lengthscale = gp.kernel_.k1.k2.length_scale
sigma_f = np.sqrt(gp.kernel_.k1.k1.constant_value)
sigma_y = np.sqrt(gp.kernel_.k2.noise_level)
y_pred, std = gp.predict(X_, return_std=True)
y_pred = y_pred.ravel()
sigma = std.ravel()

# Plot the covariance matrix
K = utils.RBFKernel(X, X, lengthscale, sigma_f) + sigma_y**2 * np.eye(n)

plt.figure(figsize=(10, 5))
ax = plt.subplot(121)
ax.imshow(K, cmap='YlGnBu')
ax.set_xlabel('Data index')
ax.set_ylabel('Data index')
cbar = plt.colorbar(ax.contourf(K, cmap='YlGnBu'))
ax.set_title('Covariance Matrix')
# Plot the eigenvalues
eigvals, eigvecs = np.linalg.eigh(K)
ax1 = plt.subplot(122)
ax1.plot(eigvals[::-1], '-')
ax1.set_xlabel('Eigenvalue index')
ax1.set_ylabel('Eigenvalue')
ax1.set_title('Eigenvalues of the Covariance Matrix')

sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]

# Calculate the cumulative sum of eigenvalues
cumulative_variance = np.cumsum(eigvals)
total_variance = np.sum(eigvals)

# Find the number of components needed to explain 95% of the variance
n_top = np.argmax(cumulative_variance / total_variance >= 0.95) + 1

print(f"Number of components needed to explain 95% of the variance: {n_top}")

# Highlight the eigenvalues that explain 95% of the variance
ax1.axvline(x=n_top, color='r', linestyle='--')
ax1.text(n_top, 0.5, f'N = {n_top}', rotation=90, verticalalignment='center')

# Select the top eigenvalues and corresponding eigenvectors
top_eigvals = eigvals[:n_top]
top_eigvecs = eigvecs[:, :n_top]

# Initialize a set to store all influential points
all_influential_points = set()

# For each of these eigenvectors, find the data points corresponding to the largest magnitude entries.
# These points are the "influential points" for the eigenvector.
for i in range(n_top):
    # Compute absolute values of the eigenvector
    abs_eigvec = np.abs(top_eigvecs[:, i])
    
    # Set a threshold (e.g., 50% of the maximum value)
    threshold = 0.99  * np.max(abs_eigvec)
    
    # Find points above the threshold
    influential_points = np.where(abs_eigvec > threshold)[0]
    
    print(f"Influential points for eigenvector {i}: {influential_points}")
    all_influential_points.update(influential_points)

# Highlight the influential points in the covariance matrix
influential_points_list = list(all_influential_points)
ax.scatter(influential_points_list, influential_points_list, color='red', s=100, marker='x')

# Plot
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax1.plot(X_, y_true, 'r:', label=r'$f(x) = \sin(x)$')
ax1.plot(X, y, 'r.', markersize=10, label='Observations')
ax1.plot(X_, y_pred, 'b-', label='Prediction')
ax1.fill_between(X_.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$f(x)$')
ax1.set_ylim(-1.5, 1.5)
ax1.legend(loc='upper left')
ax1.legend()

# Fit a GP to the influential points
X_subset = X[influential_points_list]
y_subset = y[influential_points_list]

# GP
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
gp_subset = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
tstart = time.time()
gp_subset.fit(X_subset, y_subset)
print(f"Training time subset GP: {time.time() - tstart:.2f}s")
y_pred_subset, sigma_subset = gp_subset.predict(X_, return_std=True)
y_pred_subset = y_pred_subset.ravel()
sigma_subset = sigma_subset.ravel()

ax4 = fig.add_subplot(212)
ax4.plot(X_, y_true, 'r:', label=r'$f(x) = \sin(x)$')
ax4.plot(X_subset, y_subset, 'g.', markersize=10, label='Observations')
ax4.plot(X_, y_pred_subset, 'b-', label='Prediction')
ax4.fill_between(X_.ravel(), y_pred_subset - 1.96 * sigma_subset, y_pred_subset + 1.96 * sigma_subset, alpha=0.2)
ax4.set_xlabel('$x$')
ax4.set_ylabel('$f(x)$')
ax4.set_ylim(-1.5, 1.5)
ax4.legend(loc='upper left')
ax4.legend()

# Plot the cumulative variance explained

plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance / total_variance, '-o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative variance explained')
plt.title('Cumulative variance explained by PCA components')
plt.axvline(x=n_top, color='r', linestyle='--')
plt.text(n_top, 0.5, f'N = {n_top}', rotation=90, verticalalignment='center')

# Print the sorted unique influential points
print(f"Sorted unique influential points: {sorted(list(all_influential_points))}")
print(f"Number of influential points: {len(all_influential_points)}")

plt.show()