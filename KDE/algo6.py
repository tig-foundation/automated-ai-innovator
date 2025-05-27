
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    Improved Gaussian kernel density estimation using Scott's rule for bandwidth selection.
    
    :param np.ndarray train_xs: Training data points (num_train, dims)
    :param np.ndarray test_xs: Test data points (num_test, dims)
    :returns: Scalar log-likelihood of test data
    """
    _log_twopi = np.log(2 * np.pi)

    def log_prob_Gaussian(ys, mus, sigmas):
        """
        Log probability density for a Gaussian distribution.
        
        :param np.ndarray ys: Points to evaluate
        :param np.ndarray mus: Means of the Gaussians (..., dims)
        :param np.ndarray sigmas: Standard deviations (..., dims)
        """
        ll = -0.5 * (_log_twopi + 2 * np.log(sigmas + 1e-12) 
                     + ((ys - mus) ** 2) / (sigmas ** 2 + 1e-10))
        return ll.sum(-1)

    def log_prob_Gaussian_mixture(x_eval, mus, sigmas):
        """
        Log density estimation for Gaussian Mixture Model.

        :param np.ndarray x_eval: Test points (..., num_pts, dims)
        :param np.ndarray mus: Gaussian centers (num_train, dims)
        :param np.ndarray sigmas: Bandwidth for each dimension (1, dims)
        :return: Log density (..., num_test)
        """
        ll = log_prob_Gaussian(x_eval[:, None, :], mus[None, :, :], sigmas)
        return logsumexp(ll, axis=1) - np.log(num_train)

    num_train, dims = train_xs.shape
    
    # Scott's rule for bandwidth selection
    std_devs = np.std(train_xs, axis=0, keepdims=True)
    scott_bws = std_devs * (num_train ** (-1 / (dims + 4)))

    return log_prob_Gaussian_mixture(test_xs, train_xs, scott_bws).mean()
