
import numpy as np
from scipy.special import logsumexp


def algorithm_func(train_xs, test_xs):
    """
    Implements Gaussian Kernel Density Estimation (KDE) using Silverman's rule for bandwidth selection.

    :param train_xs: Training data, shape (num_train, dims)
    :param test_xs: Test data, shape (num_test, dims)
    :return: Estimated log likelihood of the test data (scalar).
    """
    _log_twopi = np.log(2 * np.pi)

    def log_prob_gaussian(x, mean, std):
        """
        Compute log probability under Gaussian for data points `x`.
        :param x: Points to evaluate (shape ..., dims)
        :param mean: Gaussian mean (same shape as `x`)
        :param std: Gaussian std dev (same shape as `x`)
        :return: Log likelihood evaluation (shape ...)
        """
        log_density = -_log_twopi * x.shape[-1] / 2 - 0.5 * np.sum(((x - mean) / std) ** 2, axis=-1)
        return log_density - np.sum(np.log(std), axis=-1)

    def log_prob_kde(x_eval, train_xs, bandwidths):
        """
        Compute Kernel Density Estimate log probability for data points `x_eval`.
        :param x_eval: Points to evaluate KDE (shape ..., dims)
        :param train_xs: Training data (shape num_train, dims)
        :param bandwidths: Bandwidths (shape dims)
        :return: Log likelihood evaluation (shape ...)
        """
        distances = x_eval[:, None, :] - train_xs[None, :, :]
        log_gaussians = log_prob_gaussian(distances, mean=0, std=bandwidths)
        return logsumexp(log_gaussians, axis=1) - np.log(len(train_xs))

    # Silverman's rule for bandwidth selection
    num_train, dims = train_xs.shape
    std_devs = np.std(train_xs, axis=0, keepdims=True)
    bandwidths = 0.9 * std_devs * (num_train ** (-1 / (dims + 4)))

    return np.mean(log_prob_kde(test_xs, train_xs, bandwidths))
