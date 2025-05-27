
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    This algorithm implements Gaussian kernel density estimation using Silverman's bandwidth heuristic and Gaussian kernels.
    :param np.ndarray train_xs: training data points (num_train, dims)
    :param np.ndarray test_xs: test data points (num_test, dims)
    :return: log likelihood estimate of test data
    """
    # Constants for Gaussian probability
    _log_twopi = np.log(2 * np.pi)

    def log_prob_Gaussian(ys, mus, sigmas):
        """
        Computes log probability of data points under a Gaussian density.
        :param np.ndarray ys: evaluation points (..., D)
        :param np.ndarray mus: mean points (..., D)
        :param np.ndarray sigmas: standard deviations (..., D)
        :return: log probability density values
        """
        ll = -0.5 * _log_twopi * ys.shape[-1] - np.log(sigmas + 1e-12).sum(-1) - 0.5 * (((ys - mus) / (sigmas + 1e-12)) ** 2).sum(-1)
        return ll

    def log_prob_Gaussian_mixture(xs, probs, means, scales):
        """
        Computes log probability for a mixture of Gaussians.
        :param np.ndarray xs: evaluation points (..., D)
        :param np.ndarray probs: mixture weights (N,)
        :param np.ndarray means: mixture means (N, D)
        :param np.ndarray scales: mixture scales (N, D)
        :return: log density values (...,)
        """
        log_weights = np.log(probs)
        log_densities = log_prob_Gaussian(xs[..., None, :], means, scales)  # (..., N)
        return logsumexp(log_weights + log_densities, axis=-1)

    # Compute Silverman's bandwidth
    num_train, dims = train_xs.shape
    std_devs = train_xs.std(axis=0, keepdims=True)
    silverman_bws = 0.9 * std_devs * num_train ** (-1 / 5)

    bandwidths = np.broadcast_to(silverman_bws, train_xs.shape)
    weights = np.full(num_train, 1.0 / num_train)  # Uniform weights

    log_likelihood_test = log_prob_Gaussian_mixture(test_xs, weights, train_xs, bandwidths)
    return log_likelihood_test.mean()
