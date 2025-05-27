
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    This algorithm implements Gaussian kernel density estimation using Silverman's rule.

    :param np.ndarray train_xs: training data points (num_train, dims)
    :param np.ndarray test_xs: test data points (num_test, dims)
    :return: Log likelihoods (scalar)
    """
    # Precomputed constant to avoid repetitive calculation
    _log_twopi = np.log(2 * np.pi)

    # Function to compute log probability of Gaussian
    def log_prob_Gaussian(ys, mus, sigmas):
        ll = -_log_twopi * ys.shape[-1] / 2 - np.sum(np.log(sigmas + 1e-12), axis=-1) \
             - np.sum(0.5 * ((ys - mus) ** 2) / (sigmas ** 2 + 1e-12), axis=-1)
        return ll

    # Silverman bandwidth computation
    num_train, dims = train_xs.shape
    std_devs = train_xs.std(axis=0, keepdims=True)
    silverman_bws = 0.9 * std_devs * num_train ** (-1 / 5)

    # Weights, means, and standard deviations for kernel density estimation
    ws = np.ones(num_train) / num_train
    mus = train_xs
    sigmas = silverman_bws

    # Log probability computation across the test points
    log_ws = np.log(ws)
    ll = log_prob_Gaussian(test_xs[:, None, :], mus[None, :, :], sigmas[None, :, :])  # Broadcasting for kernel density
    log_likelihoods = logsumexp(log_ws + ll, axis=1)

    return np.mean(log_likelihoods)
