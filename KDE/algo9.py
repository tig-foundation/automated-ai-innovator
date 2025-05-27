
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    Kernel density estimation using Gaussian kernels with Silverman's rule for bandwidth selection.
    
    :param np.ndarray train_xs: Training data points (num_train, dims).
    :param np.ndarray test_xs: Test data points (num_test, dims).
    :return: Scalar log likelihood of the test data.
    """
    _log_twopi = np.log(2 * np.pi)
    
    def log_prob_gaussian(ys, mus, sigmas):
        """
        Compute Gaussian log probabilities.
        
        :param np.ndarray ys: Points at which density is evaluated (..., D).
        :param np.ndarray mus: Means (..., D).
        :param np.ndarray sigmas: Standard deviations (..., D).
        :return: Log density (...).
        """
        ll = -0.5 * (_log_twopi * ys.shape[-1] + np.log(sigmas.prod(axis=-1) + 1e-12) +
                     np.sum(((ys - mus) ** 2) / (sigmas ** 2 + 1e-10), axis=-1))
        return ll
    
    num_train, dims = train_xs.shape
    std_devs = train_xs.std(axis=0, keepdims=True)
    silverman_bw = 1.06 * std_devs * num_train ** (-1 / (dims + 4))  # Slightly adjusted heuristic for bandwidth
    
    log_likelihoods = []
    for x in test_xs:
        ll = log_prob_gaussian(x, train_xs, silverman_bw)
        log_likelihoods.append(logsumexp(ll) - np.log(len(ll)))
    
    return np.mean(log_likelihoods)
