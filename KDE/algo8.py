
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    Revised algorithm implementing Gaussian kernel density estimation with Silverman's rule for bandwidth selection
    and optimized code for complexity reduction.
    
    :param np.ndarray train_xs: training data points (num_train, dims)
    :param np.ndarray test_xs: test data points (num_test, dims)
    :return: scalar log-likelihood estimate for the test data
    """
    def log_prob_Gaussian(ys, mus, sigmas):
        """
        Compute element-wise Gaussian log density.

        :param np.ndarray ys: inputs for density evaluation (..., D)
        :param np.ndarray mus: mean of Gaussian (..., D)
        :param np.ndarray sigmas: standard deviation (..., D)
        :return: log density values (..., D)
        """
        diff = ys - mus
        return -0.5 * np.sum(np.log(2 * np.pi * sigmas**2) + (diff**2) / (sigmas**2), axis=-1)
    
    def log_prob_Gaussian_mixture(xs, ws, mus, sigmas):
        """
        Compute log probabilities for a Gaussian Mixture Model.

        :param np.ndarray xs: points for evaluation (..., num_pts, D)
        :param np.ndarray ws: weights for components (num_pts,)
        :param np.ndarray mus: means (num_pts, D)
        :param np.ndarray sigmas: standard deviations (num_pts, D)
        :return: log mixture density per point (..., num_pts)
        """
        ll = log_prob_Gaussian(xs[:, None, :], mus, sigmas)  # (..., num_pts)
        return logsumexp(np.log(ws) + ll, axis=-1)
    
    # Use Silverman's rule to determine bandwidth
    num_train, dims = train_xs.shape
    bandwidth = 1.06 * train_xs.std(axis=0) * num_train ** (-1 / (dims + 4))
    weights = np.ones(num_train) / num_train  # Uniform weights for training points
    
    # Estimate log-likelihood for test points
    return np.mean(log_prob_Gaussian_mixture(test_xs, weights, train_xs, bandwidth))
