import numpy as np
from numpy.linalg import inv, pinv, cholesky, det
from scipy.linalg import solve_triangular
from scipy.stats import norm

def kernel(X1: np.ndarray, 
           X2: np.ndarray, 
           sigma_f: float=1.0, 
           lengthscale: float=1.0) -> np.ndarray:
    """
    Exponentiated Quadratic Kernel
    (https://peterroelants.github.io/posts/gaussian-process-kernels/#Exponentiated-quadratic-kernel)
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    spatiodist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

    RBF = np.exp(-0.5 * spatiodist / lengthscale**2)

    C = sigma_f**2

    return C * RBF

def posterior(X_eval: np.ndarray, 
              X_train: np.ndarray, 
              Y_train: np.ndarray,
              sigma_f: float=1.0, 
              sigma_l: float=1.0, 
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

    k = kernel(X_train, X_train, sigma_l=sigma_l, sigma_f=sigma_f) + sigma_y**2 * np.eye(len(X_train))
    k_s = kernel(X_train, X_eval, sigma_l=sigma_l, sigma_f=sigma_f)
    k_ss = kernel(X_eval, X_eval, sigma_l=sigma_l, sigma_f=sigma_f)
    k_inv = inv(k)

    mu_s = k_s.T.dot(k_inv).dot(Y_train)
    cov_s = (k_ss - k_s.T.dot(k_inv).dot(k_s)) # + sigma_y**2 * np.eye(len(X_eval))
    
    return mu_s, cov_s

def nll_fn(X_train, Y_train, naive=False):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        naive: if True use a naive implementation of Eq. (11), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """

    Y_train = Y_train.ravel()

    def nll_naive(theta):
        k = kernel(X_train, X_train, lengthscale=theta[0], sigma_f=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))

        return 0.5 * np.log(det(k)) + \
               0.5 * Y_train.dot(inv(k).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
        
    def nll_stable(theta):      
        k = kernel(X_train, X_train, lengthscale=theta[0], sigma_f=theta[1]) + \
            theta[2]**2 * np.eye(len(X_train))
        L = cholesky(k)
        
        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.dot(S2) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable
    

