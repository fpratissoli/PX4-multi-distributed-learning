import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from matplotlib.gridspec import GridSpec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
import time
import utilities as utils
warnings.filterwarnings("ignore")

class robot:
    def __init__(self, id):
        self._id = id
        self.dataset = np.empty((0, 3))
        self.hyps = np.array([1, 1, 1], dtype=np.float128) # lengthscale, sigma_f, sigma_y
        self.neighbors_hyps = np.empty((0, 3), dtype=np.float128)
        self._p = np.array([0, 0, 0], dtype=np.float128)
        self._mean = np.array([0, 0], dtype=np.float128) # mu (mean)
        self._cov = np.array([0, 0], dtype=np.float128) # sigma**2 (variance)
        self._cov_rec = np.array([0, 0], dtype=np.float128) # sigma**-2 (variance)
        self._w_mu = np.array([0, 0], dtype=np.float128) # w_mu (mean)
        self._w_cov = np.array([0, 0], dtype=np.float128)
        self._tmp_w_mu = np.array([0, 0], dtype=np.float128)
        self._tmp_w_cov = np.array([0, 0], dtype=np.float128)
        self._position = np.array([0, 0], dtype=np.float128)
        self._neighbors = []

    def set_hyps(self, hyps):
        self.hyps = hyps
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def set_neighbors_hyps(self, neighbors_hyps):
        self.neighbors_hyps = neighbors_hyps

    def get_neighbors_hyps(self):
        return self.neighbors_hyps
    
    def get_hyps(self):
        return self.hyps
    
    def get_dataset(self):
        return self.dataset
    
    @property
    def id(self):
        return self._id

    @property
    def lengthscale(self):
        return self.hyps[0]

    @property
    def sigma_f(self):
        return self.hyps[1]
    
    @property
    def sigma_y(self):
        return self.hyps[2]
    
    @property
    def p(self):
        return self._p
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def cov(self):
        return self._cov
    
    @property
    def cov_rec(self):
        return self._cov_rec
    
    @property
    def position(self):
        return self._position
    
    @property
    def neighbors(self):
        return self._neighbors
    
    @property
    def w_mu(self):
        return self._w_mu
    
    @property
    def w_cov(self):
        return self._w_cov
    
    @property
    def tmp_w_mu(self):
        return self._tmp_w_mu
    
    @property
    def tmp_w_cov(self):
        return self._tmp_w_cov

    @p.setter
    def p(self, value):
        self._p = value

    @mean.setter
    def mean(self, value):
        self._mean = value

    @cov.setter
    def cov(self, value):
        self._cov = value
    
    @cov_rec.setter
    def cov_rec(self, value):
        self._cov_rec = value

    @position.setter
    def position(self, value):
        self._position = value

    @neighbors.setter
    def neighbors(self, value):
        self._neighbors = value

    @w_mu.setter
    def w_mu(self, value):
        self._w_mu = value

    @w_cov.setter
    def w_cov(self, value):
        self._w_cov = value

    @tmp_w_mu.setter
    def tmp_w_mu(self, value):
        self._tmp_w_mu = value

    @tmp_w_cov.setter
    def tmp_w_cov(self, value):
        self._tmp_w_cov = value

def plot_ax(ax, data, x1, x2, title, cmap='viridis'):
    """ Plots the data. """
    field_p = ax.contourf(x1, x2, data.reshape(len(x1), len(x2)), cmap=cmap)
    # Divide the plot into 4 regions
    ax.plot([5, 5], [0, 10], 'k-', zorder=1, alpha=0.5)
    ax.plot([0, 10], [5, 5], 'k-', zorder=1, alpha=0.5)
    # Colorbar
    # cbar = plt.colorbar(field_p, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

def plot_drone(ax, x, y, orientation=0, color='k'):
    """ Plots the drone position. """
    # Draw the drone body with the orientation
    p1 = [x + 0.25*cos(orientation), y + 0.25*sin(orientation)]
    p2 = [x - 0.25*sin(orientation), y + 0.25*cos(orientation)]
    p3 = [x - 0.25*cos(orientation), y - 0.25*sin(orientation)]
    p4 = [x + 0.25*sin(orientation), y - 0.25*cos(orientation)]

    # Draw the drone body frame
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], c=color, zorder=2)
    ax.plot([p2[0], p4[0]], [p2[1], p4[1]], c=color, zorder=2)

    # Draw the drone propellers
    ax.scatter(p1[0], p1[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p2[0], p2[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p3[0], p3[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p4[0], p4[1], c=color, marker='o', s=200, zorder=2)

def RBFKernel(X1: np.ndarray, 
           X2: np.ndarray,
           lengthscale: float=1.0,
           sigma_f: float=1.0) -> np.ndarray:
    """
    Exponentiated Quadratic Kernel
    (https://peterroelants.github.io/posts/gaussian-process-kernels/#Exponentiated-quadratic-kernel)
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

    RBF = np.exp(- 0.5 * sqdist / lengthscale**2)

    C = sigma_f**2

    return C * RBF

def log_likelihood_grad(robot):
    """ 
    Computes the gradient of the log-likelihood with respect to 
    the hyperparameters: lengthscale, sigma_f, and sigma_y.
    """
    X = robot.get_dataset()[:, :2]
    y = robot.get_dataset()[:, 2]
    y = np.atleast_2d(y).T
    
    # Calculate the RBF kernel
    K = RBFKernel(X, X, sigma_f=robot.sigma_f, lengthscale=robot.lengthscale)
    
    # Add noise variance term
    C_theta = K + robot.sigma_y**2 * np.eye(len(X)) 
    
    C_theta_inv = np.linalg.solve(C_theta, np.eye(len(X)))
    
    # Compute the squared distance matrix
    sqdist = np.sum(X**2, 1, dtype=np.float128).reshape(-1, 1) + np.sum(X**2, 1, dtype=np.float128) - 2 * np.dot(X, X.T)
    
    # Gradients of the covariance matrix with respect to hyperparameters
    dC_lengthscale = K * sqdist / (robot.lengthscale**3)
    dC_theta_f = 2 * K / robot.sigma_f
    dC_theta_y = np.eye(len(X)) * 2 * robot.sigma_y

    # Compute the gradient of the log-likelihood
    common_term = C_theta_inv - (C_theta_inv @ y @ y.T @ C_theta_inv)
    dNLL_lengthscale = 0.5 * np.trace(common_term @ dC_lengthscale, dtype=np.float128)
    dNLL_sigma_f = 0.5 * np.trace(common_term @ dC_theta_f, dtype=np.float128)
    dNLL_sigma_y = 0.5 * np.trace(common_term @ dC_theta_y, dtype=np.float128)

    return np.array([dNLL_lengthscale, dNLL_sigma_f, dNLL_sigma_y], dtype=np.float128)

# Generate data
np.random.seed(0)

area_size = 40
d_field_ = 1
x1_ = np.arange(0, area_size + d_field_, d_field_)
x2_ = np.arange(0, area_size + d_field_, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Generate random means
peaks = 8 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 5
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means)
Z = Z.reshape(len(x1_), len(x2_))

n_robots = 6
robots = []
for i in range(n_robots):
    robots.append(robot(i))

grid_width = 3
grid_height = 2

region_width = area_size / grid_width  # Width of each region
region_height = area_size / grid_height  # Height of each region

regions = []
for i in range(grid_width * grid_height):
    x1_min = (i % grid_width) * region_width
    x1_max = x1_min + region_width
    x2_min = (i // grid_width) * region_height
    x2_max = x2_min + region_height
    regions.append([x1_min, x1_max, x2_min, x2_max])

# Place the robots in their regions midpoints
for i, robot in enumerate(robots):
    x1_min, x1_max, x2_min, x2_max = regions[i]
    robot.position = np.array([x1_min + region_width / 2, x2_min + region_height / 2])

# Find the robots' neighbors
for i, robot in enumerate(robots):
    for j, other_robot in enumerate(robots):
        if i != j:
            if np.linalg.norm(robot.position - other_robot.position) <= 1.5 * region_width:
                robot.neighbors.append(other_robot)

# robotA = robots[4]
# robotB = robots[1]
# robotA.neighbors.remove(robotB)
# robotB.neighbors.remove(robotA)

# Update adjacency matrix and max_degree
A = np.zeros((n_robots, n_robots))
for i, robot in enumerate(robots):
    for j, other_robot in enumerate(robots):
        if i != j:
            if other_robot in robot.neighbors:
                A[i, j] = 1
            else:
                A[i, j] = 0

max_degree = np.max(np.sum(A, axis=1))

# For each drone sample some points in its region
n_points = 100
noise = 0.1

for i, robot in enumerate(robots):
    x1_min, x1_max, x2_min, x2_max = regions[i]
    x1 = np.random.uniform(x1_min, x1_max, n_points)
    x2 = np.random.uniform(x2_min, x2_max, n_points)
    X = np.vstack([x1, x2]).T
    y = utils.gmm_pdf_array(x1, x2, sigma, means) + np.random.normal(0, noise, n_points)
    robot.set_dataset(np.hstack([X, np.atleast_2d(y).T]))

# Plot the datasets with different colors
fig, ax_main = plt.subplots()
ax_main.set_aspect('equal')
field_p = ax_main.contour(x1_, x2_, Z, cmap='YlGnBu', zorder=0)
for robot in robots:
    X = robot.get_dataset()[:, :2]
    ax_main.scatter(X[:, 0], X[:, 1], marker='x', alpha=0.2)
    ax_main.scatter(robot.position[0], robot.position[1], c='r', marker='o', s=100)
    ax_main.text(robot.position[0], robot.position[1], f"{robot.id}", fontsize=12)

for robot in robots:
    for neighbor in robot.neighbors:
        ax_main.plot([robot.position[0], neighbor.position[0]], [robot.position[1], neighbor.position[1]], 'k-', alpha=0.5)
plt.grid(alpha=0.2)

total_dataset = np.empty((0, 3))
for robot in robots:
    total_dataset = np.vstack([total_dataset, robot.get_dataset()])

x_train, y_train = total_dataset[:, :2], total_dataset[:, 2]
y_train = np.atleast_2d(y_train).T
kernel = ConstantKernel() * RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b")
s_time = time.time()
gp.fit(x_train, y_train)
e_time = time.time()

print(gp.kernel_)

lengthscale_full = gp.kernel_.get_params()['k1__k2__length_scale']
sigma_f_full = np.sqrt(gp.kernel_.get_params()['k1__k1__constant_value'])
sigma_y_full = np.sqrt(gp.kernel_.get_params()['k2__noise_level'])

print(f"Hyperparameters: lengthscale - {lengthscale_full} | sigma_f - {sigma_f_full} | sigma_y - {sigma_y_full}")

# Reset the hyperparameters
for robot in robots:
    robot.set_hyps(np.array([1, 1, 1])) # lengthscale, sigma_f, sigma_y

""" DEC-gapx-GP (Kontoudis et al.) """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3
old_hypers = np.zeros((n_robots, 3)) # Initialize hypers with zeros

s_time = time.time()

# Vector to store the results
results = []
p_s = []
first_terms = []
second_terms = []
third_terms = []
for robot in robots:
    results.append([])
    p_s.append([])
    first_terms.append([])
    second_terms.append([])
    third_terms.append([])

for s in range(s_end_DEC_gapx):
    print(f"*** {s} ***")
    tmp_hyps = np.empty((0, 3), dtype=np.float128) # Temporary store the new hyperparameters
    for robot in robots:
        # Take the neighbors' hyperparameters
        neighbors_hyps = np.empty((0, 3), dtype=np.float128)
        for other_robot in robot.neighbors:
            neighbors_hyps = np.vstack([neighbors_hyps, other_robot.get_hyps()], dtype=np.float128)
        n_neighbors = neighbors_hyps.shape[0]

        # Duals (30a) (Consensus)
        sum = np.array([0, 0, 0], dtype=np.float128)
        for k in range(n_neighbors):
            sum = sum + (robot.get_hyps() - neighbors_hyps[k])
        robot.p = robot.p + rho * sum

        # Primal (34b) (ADMM)
        first_term = rho * np.sum(neighbors_hyps, axis=0, dtype=np.float128)
        second_term = log_likelihood_grad(robot)
        third_term = (ki + n_neighbors * rho) * robot.get_hyps()
        res = (1 / (ki + 2 * n_neighbors * rho)) * (first_term - second_term + third_term - robot.p)
        tmp_hyps = np.vstack([tmp_hyps, res], dtype=np.float128)
        
    old_hypers = np.copy([robot.hyps for robot in robots])
    for i, robot in enumerate(robots):
        robot.set_hyps(tmp_hyps[i])

print("Hyperparameters after DEC-gapx-GP:")
for robot in robots:
    print(f"Robot {robot.id} - {robot.get_hyps()}")

# Plot the new predictions for each robot
fig, axes = plt.subplots(2, 3)
for i in range(n_robots):
    ax = axes[i // 3, i % 3]
    ax.set_aspect('equal')
    x_train, y_train = robots[i].get_dataset()[:, :2], robots[i].get_dataset()[:, 2]
    y_train = np.atleast_2d(y_train).T
    mu, std = utils.posterior(mesh, x_train, y_train, lengthscale=robots[i].lengthscale, sigma_f=robots[i].sigma_f, sigma_y=robots[i].sigma_y)
    ax.contour(x1_, x2_, mu.reshape(len(x1_), len(x2_)), cmap='YlGnBu')

fig.suptitle("No DEC-PoE")

""" Implement the filtering algorithm on the dataset """

""" DEC-PoE-GP (Kontoudis et al.) """
eps = 1 / max_degree
eps = eps / 2
beta = 1 / n_robots
s_end_DAC = 1000

# Compute the local predictions
for robot in robots:
    x_train, y_train = robot.get_dataset()[:, :2], robot.get_dataset()[:, 2]
    y_train = np.atleast_2d(y_train).T
    mu, cov = utils.posterior(mesh, x_train, y_train, lengthscale=robot.lengthscale, sigma_f=robot.sigma_f, sigma_y=robot.sigma_y)
    robot.mean = np.reshape(mu, (len(x1_), len(x2_))) # 41 x 41
    robot.cov =  np.reshape(np.diag(cov), (len(x1_), len(x2_))) # 41 x 41
    robot.cov_rec = np.reshape(1/np.diag(cov), (len(x1_), len(x2_))) # 41 x 41

# Initialize weights
for robot in robots:
    robot.w_mu = beta * robot.cov_rec * robot.mean
    robot.w_cov = beta * robot.cov_rec

shape = (len(x1_), len(x2_))
sTime = time.time()
for s in range(s_end_DAC):
    sum_mu_diff = np.zeros(shape, dtype=np.float128)
    sum_cov_diff = np.zeros(shape, dtype=np.float128)
    for robot in robots:
        neighbors_w_mu = np.array([other_robot.w_mu for other_robot in robot.neighbors])
        neighbors_w_cov = np.array([other_robot.w_cov for other_robot in robot.neighbors])

        # DAC 1 (Mean)
        sum_mu_diff = np.sum(neighbors_w_mu, axis=0) - robot.w_mu * len(robot.neighbors)
        robot.tmp_w_mu = robot.w_mu + eps * sum_mu_diff

        # DAC 2 (Covariance)
        sum_cov_diff = np.sum(neighbors_w_cov, axis=0) - robot.w_cov * len(robot.neighbors)
        robot.tmp_w_cov = robot.w_cov + eps * sum_cov_diff
    for robot in robots:
        robot.w_mu = robot.tmp_w_mu
        robot.w_cov = robot.tmp_w_cov

for robot in robots:
    robot.cov_rec = n_robots * robot.w_cov
    robot.mean = (1 / robot.cov_rec) * (n_robots * robot.w_mu)
print(f"Time: {time.time() - sTime}")
print("Done DEC-PoE!")

# RMSE with original GP
rmse = np.sqrt(np.mean((Z - robot.mean)**2))
print(f"RMSE: {rmse}")

# Plot the results
fig, axes = plt.subplots(2, 3)
for i in range(n_robots):
    ax = axes[i // 3, i % 3]
    ax.set_aspect('equal')
    ax.contourf(x1_, x2_, robots[i].mean, cmap='YlGnBu')
    ax.set_title(f"{robots[i].id}")
fig.suptitle("DEC-PoE")

plt.tight_layout()
plt.show()