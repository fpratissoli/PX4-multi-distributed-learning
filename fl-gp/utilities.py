#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi
from matplotlib.path import Path
from shapely.geometry import Polygon, Point
import xarray as xr
from scipy.interpolate import griddata
import time

def evaluate_points_in_field(field, points, method='linear'):
    """
    Evaluate the values at specified points in a given 2D field.

    Parameters:
    - field: 2D numpy array representing the field.
    - points: numpy array of shape (n_points, 2) representing the coordinates to evaluate.
    - method: Interpolation method (default is 'linear'). Options: 'linear', 'nearest', 'cubic'.

    Returns:
    - values: 1D numpy array of interpolated values at the specified points.
    """
    # Define the grid based on the field dimensions
    grid_x, grid_y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))

    # Flatten the grid and the field for interpolation
    coords = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    field_flat = field.flatten()

    # Interpolate the values at the specified points
    values = griddata(coords, field_flat, points, method=method)
    
    return values

def gauss_pdf_array(x, y, sigma, mean, max_val=1.0):
    """ Computes the probability density function of a 2D Gaussian distribution. """
    xt, yt = mean
    val = ((x - xt)**2 + (y - yt)**2) / (2 * sigma**2)
    return np.exp(-val) * max_val

def gmm_pdf_array(x, y, sigma, means, flag_normalize=False):
    """
    Computes the probability density function of a 2D Gaussian mixture model.

    Parameters:
        x (numpy.ndarray or float): x-coordinates of the grid or specific points.
        y (numpy.ndarray or float): y-coordinates of the grid or specific points.
        sigma (float): Standard deviation for the Gaussian components.
        means (list of tuples): List of mean positions (xt, yt) for each Gaussian component.
        flag_normalize (bool): If True, normalize the resulting field.

    Returns:
        numpy.ndarray or float: Evaluated GMM PDF over the grid or at specific points.
    """
    # Initialize the value of the field
    val = np.zeros_like(x, dtype=np.float64)

    # Compute the GMM
    for mean in means:
        xt, yt = mean
        val += np.exp(-((x - xt)**2 + (y - yt)**2) / (2 * sigma**2))

    # Normalize the field if required
    if flag_normalize:
        val /= np.sum(val)

    return val

def sense_neighbors(robots: np.ndarray) -> np.ndarray:
    """
    Sense the neighbors of each robot and update the neighbors attribute.
    """
    M = len(robots) # Agents

    for robot in robots:
        neighbors = np.array([])
        for other_robot in robots:
            if other_robot != robot:
                dist = np.linalg.norm(robot.position - other_robot.position)
                if dist <= robot.range:
                    neighbors = np.append(neighbors, other_robot)
        robot.neighbors = neighbors

    return robots

def update_A(robots: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Update the adjacency matrix of the robots.
    """
    for i, robot in enumerate(robots):
        for j, other_robot in enumerate(robots):
            if i != j:
                dist = np.linalg.norm(robot.position - other_robot.position)
                if dist <= robot.range:
                    A[i, j] = 1
                else:
                    A[i, j] = 0

    return A

def find_groups(robots: np.ndarray, A: np.ndarray) -> None:
    """
    Find the groups of robots and assign the group ID to each robot.

    Parameters:
    robots (np.ndarray): An array of Robot objects.
    A (np.ndarray): Adjacency matrix representing the connections between robots.
    """
    M = len(robots)  # Number of robots
    group_id = 1  # Group ID starts from 1

    def dfs(node, current_group_id):
        """Depth-First Search to mark all robots in the same group."""
        robots[node].group = current_group_id
        for neighbor in range(M):
            if A[node, neighbor] == 1 and robots[neighbor].group is None:
                dfs(neighbor, current_group_id)

    # Reset groups before starting
    for robot in robots:
        robot.group = None

    for i in range(M):
        if robots[i].group is None:  # If this robot hasn't been assigned to a group
            dfs(i, group_id)
            group_id += 1

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

def posterior(X_eval: np.ndarray, 
              X_train: np.ndarray, 
              Y_train: np.ndarray,
              lengthscale: float=1.0, 
              sigma_f: float=1.0, 
              sigma_y: float=1e-6) -> tuple:
    """
    Computes mean and covariance of the posterior distribution.
    
    Args:
        X_eval: Input locations to evaluate the posterior (n x d).
        X_train: Training locations (m x d).
        Y_train: Training values (m x 1).
        sigma_l: Kernel length parameter (describes the spatial correlation between points).
        sigma_f: Kernel vertical variation parameter (describes the vertical variation of the kernel).
        sigma_y: Noise parameter (describes the noise level of the data).
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n) as a tuple.
    """

    k = RBFKernel(X_train, X_train, lengthscale=lengthscale, sigma_f=sigma_f) + sigma_y**2 * np.eye(len(X_train))
    k_s = RBFKernel(X_train, X_eval, lengthscale=lengthscale, sigma_f=sigma_f)
    k_ss = RBFKernel(X_eval, X_eval, lengthscale=lengthscale, sigma_f=sigma_f)
    k_inv = np.linalg.inv(k)

    mu = k_s.T.dot(k_inv).dot(Y_train)
    cov = (k_ss - k_s.T.dot(k_inv).dot(k_s)) + sigma_y**2 * np.eye(len(X_eval))

    mu[mu < 0] = 0
    cov[cov < 0] = 0
    
    return mu, cov

def log_likelihood_grad(robot):
    """ 
    Computes the gradient of the log-likelihood with respect to 
    the hyperparameters: lengthscale, sigma_f, and sigma_y.
    """
    # X = robot.observations[:, :2]
    # y = robot.observations[:, 2]
    X = robot.dataset[:, :2]
    y = robot.dataset[:, 2]
    y = np.atleast_2d(y).T
    
    # Calculate the RBF kernel
    K = RBFKernel(X, X, sigma_f=robot.sigma_f, lengthscale=robot.lengthscale)
    
    # Add noise variance term
    C_theta = K + robot.sigma_y**2 * np.eye(len(X)) + 1e-4 * np.eye(len(X))
    
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

# Function to process each group
def process_group(robots_group, s_end_DEC_gapx, s_end_DAC, rho, ki, beta, eps, x1_, x2_, ROB_NUM):
    tmp_hyps = np.empty((len(robots_group), 3))

    # Loop over iterations
    for _ in range(s_end_DEC_gapx):
        for id, robot in enumerate(robots_group):
            # Gather neighbors' hyperparameters efficiently
            neighbors_hyps = np.empty((0, 3), dtype=np.float128)
            for other_robot in robot.neighbors:
                neighbors_hyps = np.vstack([neighbors_hyps, other_robot.hyps])
            n_neighbors = neighbors_hyps.shape[0]

            # Consensus update (Duals update - 30a)
            sum_diff = np.sum(robot.hyps - neighbors_hyps, axis=0)
            robot.p += rho * sum_diff

            # Primal update (34b) using vectorized operations
            first_term = rho * np.sum(neighbors_hyps, axis=0)
            second_term = log_likelihood_grad(robot)
            third_term = (ki + n_neighbors * rho) * robot.hyps
            tmp_hyps[id] = (1 / (ki + 2 * n_neighbors * rho)) * (first_term - second_term + third_term - robot.p)

        # Update robots' hyperparameters in place
        for i, robot in enumerate(robots_group):
            robot.hyps = tmp_hyps[i]

    for robot in robots_group:
        print(f"Robot {robot.id} has hyperparameters: {robot.hyps}")

    """ DEC-PoE """
    # Compute the local predictions
    for robot in robots_group:
        robot.update_estimate()

    # Initialize the weights
    for robot in robots_group:
        robot.w_mu = beta * robot.cov_rec * robot.mean
        robot.w_cov = beta * robot.cov_rec
                
    shape = (len(x1_), len(x2_))

    for _ in range(s_end_DAC):
        sum_mu_diff = np.zeros(shape, dtype=np.float128)
        sum_cov_diff = np.zeros(shape, dtype=np.float128)
        for robot in robots_group:
            neighbors_w_mu = np.array([other_robot.w_mu for other_robot in robot.neighbors])
            neighbors_w_cov = np.array([other_robot.w_cov for other_robot in robot.neighbors])

            # DAC 1 (Mean)
            sum_mu_diff = np.sum(neighbors_w_mu, axis=0) - robot.w_mu * len(robot.neighbors)
            robot.tmp_w_mu = robot.w_mu + eps * sum_mu_diff

            # DAC 2 (Covariance)
            sum_cov_diff = np.sum(neighbors_w_cov, axis=0) - robot.w_cov * len(robot.neighbors)
            robot.tmp_w_cov = robot.w_cov + eps * sum_cov_diff

        for robot in robots_group:
            robot.w_mu = robot.tmp_w_mu
            robot.w_cov = robot.tmp_w_cov
        
    for robot in robots_group:
        robot.cov_rec = ROB_NUM * robot.w_cov
        robot.std = np.sqrt(1 / robot.cov_rec)
        robot.mean = (1 / robot.cov_rec) * (ROB_NUM * robot.w_mu)
    print("*** Done! ***")

def voronoi_algorithm(robots_positions, BBOX, limited=False, sensRange=2):
    """
    Decentralized Bounded Voronoi Computation
    """

    points_left = robots_positions.copy()
    points_right = robots_positions.copy()
    points_down = robots_positions.copy()
    points_up = robots_positions.copy()

    points_left[:, 0] = 2 * BBOX[0] - robots_positions[:, 0]
    points_right[:, 0] = 2 * BBOX[2] - robots_positions[:, 0]
    points_down[:, 1] = 2 * BBOX[1] - robots_positions[:, 1]
    points_up[:, 1] = 2 * BBOX[3] - robots_positions[:, 1]

    points = np.vstack([robots_positions, points_left, points_right, points_down, points_up])

    # Voronoi diagram
    vor = Voronoi(points)
    vor.filtered_points = robots_positions
    vor.filtered_regions = [vor.regions[i] for i in vor.point_region[:len(robots_positions)]]
    
    return vor

def voronoi_alg_limited(robots_positions, BBOX, sensRange=2):
    """
    Efficiently computes the limited Voronoi regions for a set of robots.
    
    Parameters:
    - robots_positions: Array of robot positions (n x 2).
    - BBOX: Bounding box of the environment [xmin, ymin, xmax, ymax].
    - sensRange: Sensing range for clipping the Voronoi regions.
    
    Returns:
    - A list of clipped Voronoi regions as Shapely polygons.
    """
    
    # Mirroring points to handle boundary conditions
    points_left = robots_positions.copy()
    points_right = robots_positions.copy()
    points_down = robots_positions.copy()
    points_up = robots_positions.copy()

    points_left[:, 0] = 2 * BBOX[0] - robots_positions[:, 0]
    points_right[:, 0] = 2 * BBOX[2] - robots_positions[:, 0]
    points_down[:, 1] = 2 * BBOX[1] - robots_positions[:, 1]
    points_up[:, 1] = 2 * BBOX[3] - robots_positions[:, 1]

    points = np.vstack([robots_positions, points_left, points_right, points_down, points_up])

    # Compute Voronoi diagram for the given robot positions
    vor = Voronoi(points)
    
    # Prepare to store the limited Voronoi regions
    limited_regions = []
    
    # Iterate through each original robot point and its corresponding region
    for i in range(len(robots_positions)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        
        # Skip unbounded regions
        if -1 in region:
            continue
        
        # Get the vertices of the region
        polygon_vertices = vor.vertices[region]
        polygon = Polygon(polygon_vertices)
        
        # Define the sensing circle for the current robot
        sensing_circle = Point(robots_positions[i]).buffer(sensRange * 0.5)
        
        # Intersect the Voronoi region with the sensing circle
        limited_region = polygon.intersection(sensing_circle)
        
        # Store the result
        if not limited_region.is_empty:
            limited_regions.append(limited_region)
    
    return limited_regions

def coveragePerformanceFunc(robots_positions, sigma, means, res=50, BBOX=[0, 0, 40, 40]):
    delta = 1 / res
    H_pv = 0.0

    # Compute global Voronoi diagram
    vor = voronoi_algorithm(robots_positions, BBOX)

    for idx, region in enumerate(vor.filtered_regions):
        vertices = vor.vertices[region]
        xy = vor.filtered_points[idx]

        # Bounds of the current region
        x_inf, x_sup = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        y_inf, y_sup = np.min(vertices[:, 1]), np.max(vertices[:, 1])

        dx = (x_sup - x_inf) * delta
        dy = (y_sup - y_inf) * delta

        path = Path(vertices)

        x_vals = np.arange(x_inf, x_sup, dx)
        y_vals = np.arange(y_inf, y_sup, dy)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        for point in grid_points:
            i, j = point
            bool_val = path.contains_points([point])[0]
            if bool_val:  # Only update if the point is inside the region
                weight = gmm_pdf_array(i, j, sigma, means, flag_normalize=False)
                H_pv += (np.linalg.norm(point - xy)**2) * weight * dx * dy

    return H_pv

def coveragePerformanceFuncDataset(robots_positions, field, res=50, BBOX=[0, 0, 40, 40]):
    delta = 1 / res
    H_pv = 0.0

    # Define the grid based on the field dimensions
    grid_x, grid_y = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))

    # Flatten the grid and the field for interpolation
    coords = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    field_flat = field.flatten()

    # Compute global Voronoi diagram
    vor = voronoi_algorithm(robots_positions, BBOX)

    for idx, region in enumerate(vor.filtered_regions):
        vertices = vor.vertices[region]
        xy = vor.filtered_points[idx]

        # Bounds of the current region
        x_inf, x_sup = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        y_inf, y_sup = np.min(vertices[:, 1]), np.max(vertices[:, 1])

        # Create a grid within the bounds of the current region
        x_vals = np.arange(x_inf, x_sup, delta)
        y_vals = np.arange(y_inf, y_sup, delta)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

        # Filter grid points that are inside the current region
        path = Path(vertices)
        inside_mask = path.contains_points(grid_points)
        grid_points_inside = grid_points[inside_mask]

        if len(grid_points_inside) == 0:
            continue

        # Interpolate the field values at the points inside the region
        weights = griddata(coords, field_flat, grid_points_inside, method='linear', fill_value=0)

        # Calculate the contribution to H_pv for all points inside the region
        differences = np.linalg.norm(grid_points_inside - xy, axis=1) ** 2
        H_pv += np.sum(differences * weights) * delta**2

    return H_pv

def plot_dataset(fig, t, period, bbox, field, ax1, ax2, ax3, x1_field: np.ndarray, x2_field: np.ndarray, x1_mesh: np.ndarray, x2_mesh: np.ndarray, robots: np.ndarray, A) -> None:
    X_MIN, Y_MIN, X_MAX, Y_MAX = bbox
    size = 16
    robot_id = 0
    font = {'family': 'serif',
            'size': size}
    plt.rcParams.update({'font.family': 'serif', 'font.size': size})
    
    # Set the title of the figure
    fig.suptitle(f"Time Step: {t}", fontsize=16, color="black", family="serif", weight="bold", x=0.5, y=0.9)
    delta_axis = 1
    for ax in [ax1, ax2, ax3]:
        ax.clear()
        ax.axis("equal")
        
        # Set the font for the axis labels
        ax.set_xlim(X_MIN - delta_axis, X_MAX + delta_axis)
        ax.set_ylim(Y_MIN - delta_axis, Y_MAX + delta_axis)
        ax.set_xticks(np.arange(X_MIN, X_MAX + delta_axis, 10))
        ax.set_yticks(np.arange(Y_MIN, Y_MAX + delta_axis, 10))
        ax.set_aspect('equal', adjustable='box')
        
        # Plot the boundary
        ax.plot([X_MIN, X_MIN], [Y_MIN, Y_MAX], 'k-', lw=2)
        ax.plot([X_MIN, X_MAX], [Y_MIN, Y_MIN], 'k-', lw=2)
        ax.plot([X_MAX, X_MAX], [Y_MIN, Y_MAX], 'k-', lw=2)
        ax.plot([X_MIN, X_MAX], [Y_MAX, Y_MAX], 'k-', lw=2)
        
        # Set ticks font and size
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(size)
        #     tick.label.set_fontname('serif')
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(size)
        #     tick.label.set_fontname('serif')
        
        ax.grid(alpha=0.2)

    col = "black"
    ax1.set_title("Field and Robots", fontdict=font)
    # Parula colormap
    # cmap = LinearSegmentedColormap.from_list('parula', cm_data)
    original_field = ax1.contour(x1_mesh, x2_mesh, field, cmap="YlGnBu", extend='both')
    for i, robot in enumerate(robots):
        ax1.scatter(robot.position[0], robot.position[1], color=col, s=30)
        ax1.text(robot.position[0], robot.position[1] + 2.0, f"{i}", fontsize=14, ha='center', va='center', color=col, fontfamily="serif", fontweight="bold")
        region = robot.diagram
        ax1.plot(region[:, 0], region[:, 1], color=col, lw=2)
        # Plot the centroid
        ax1.scatter(robot.centroid[0], robot.centroid[1], color=col, s=10)

    # Plot a line between the robots that are neighbors using the adjacency matrix and avoiding duplicating lines
    # for i in range(len(robots)):
    #     for j in range(i + 1, len(robots)):  # Start from i + 1 to avoid duplicates
    #         if A[i, j] == 1:
    #             ax1.plot([robots[i].position[0], robots[j].position[0]], [robots[i].position[1], robots[j].position[1]], color="blue", lw=2, alpha=0.7, linestyle='--')
    # Plot the observations
    # for robot in robots:
    #     X = robot.get_dataset()[:, :2]
    #     y = robot.get_dataset()[:, 2]
    #     y = np.atleast_2d(y).T
    #     ax1.scatter(X[:, 0], X[:, 1], marker='x', alpha=0.2)

    mu = robots[robot_id].mean
    std = robots[robot_id].std
    
    ax2.set_title(f"Posterior Mean ({robot_id})", fontdict=font)
    post_mean = ax2.contourf(x1_mesh, x2_mesh, mu, cmap="YlGnBu", extend='both')
    
    ax3.set_title(f"Posterior Variance ({robot_id})", fontdict=font)
    post_var = ax3.contourf(x1_mesh, x2_mesh, std, cmap="gray", extend='both')
    
    NBINS = 4
    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes("bottom", size="5%", pad=0.5)
    # cbar1 = plt.colorbar(original_field, cax=cax1, orientation="horizontal", format="%.1f")
    # cbar1.ax.tick_params(rotation=90)
    # tick_locator = plt.MaxNLocator(nbins=NBINS)
    # cbar1.locator = tick_locator
    # cbar1.ax.yaxis.label.set_fontsize(size)
    # cbar1.ax.yaxis.label.set_fontname('serif')
    # cbar1.update_ticks()
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.5)
    cbar2 = plt.colorbar(post_mean, cax=cax2, orientation="horizontal", format="%.1f")
    cbar2.ax.tick_params(rotation=90)
    tick_locator = plt.MaxNLocator(nbins=NBINS)
    cbar2.locator = tick_locator
    cbar2.ax.yaxis.label.set_fontsize(size)
    cbar2.ax.yaxis.label.set_fontname('serif')
    cbar2.update_ticks()
    
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("bottom", size="5%", pad=0.5)
    cbar3 = plt.colorbar(post_var, cax=cax3, orientation="horizontal", format="%.1f")
    cbar3.ax.tick_params(rotation=90)
    tick_locator = plt.MaxNLocator(nbins=NBINS)
    cbar3.locator = tick_locator
    cbar3.ax.yaxis.label.set_fontsize(size)
    cbar3.ax.yaxis.label.set_fontname('serif')
    cbar3.update_ticks()

    # # For every robot in the simulation plot the mean and variance of the GP
    # for i in range(len(robots)):
    #     mu = robots[i].mean
    #     std = robots[i].std

    #     ax1 = axes[i, 0]
    #     ax2 = axes[i, 1]

    #     ax1.set_title(f"Posterior Mean ({i})", fontdict=font)
    #     post_mean = ax1.contourf(x1_mesh, x2_mesh, mu, cmap="YlGnBu", extend='both')

    #     ax2.set_title(f"Posterior Variance ({i})", fontdict=font)
    #     post_var = ax2.contourf(x1_mesh, x2_mesh, std, cmap="gray", extend='both')
        
    
    plt.pause(0.01)
    print(f"Time step: {t}")
    # if t in [1, 2, 3, 99, 100, 110, 120, 150, 200]:
    # plt.savefig(f"pictures/TRO/novf_simple{t}.pdf", bbox_inches='tight', format='pdf', dpi=300)
    # plt.show()
    # plt.savefig(f"frames/static/frame_{t}.png", bbox_inches='tight', format='png', dpi=300)
    # plt.savefig(f"frames/webots2/frame_{t}.png", bbox_inches='tight', format='png', dpi=300)

    if t != period:
        # cbar1.remove()
        cbar2.remove()
        cbar3.remove()