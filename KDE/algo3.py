
import numpy as np
from scipy.special import logsumexp


def algorithm_func(train_xs, test_xs):
    """
    Improved Gaussian kernel density estimation with bandwidth selection
    using Scott's rule, aiming for better test likelihood scores.
    """
    _log_twopi = np.log(2 * np.pi)

    def log_prob_Gaussian(ys, mus, sigmas):
        """Compute log density for Gaussian distribution."""
        ll = -0.5 * (np.log(2 * np.pi * (sigmas**2)) + (ys - mus)**2 / (sigmas**2))
        return ll.sum(-1)

    def log_prob_Gaussian_mixture(x_eval, ws, mus, sigmas):
        """Compute log density for Gaussian mixture."""
        log_ws = np.log(ws)
        ll = log_prob_Gaussian(x_eval[..., None, :], mus, sigmas)  # (..., num_test, num_train)
        return logsumexp(log_ws + ll, axis=-1)

    # Scott's rule for bandwidth selection
    num_train, dims = train_xs.shape
    cov_matrix = np.cov(train_xs.T, bias=True)
    bandwidths = (cov_matrix.diagonal() ** 0.5) * num_train ** (-1. / (dims + 4))
    
    ws = np.ones((1, num_train))
    mus = train_xs[None, ...]
    sigmas = bandwidths[None, ...]
    
    return log_prob_Gaussian_mixture(test_xs[:, None, :], ws, mus, sigmas)[:, 0]
