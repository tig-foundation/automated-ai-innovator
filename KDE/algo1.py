
import numpy as np
from scipy.special import logsumexp


def algorithm_func(train_xs, test_xs):
    """
    This algorithm implements Gaussian kernel density estimation using Silverman

    :param np.ndarray train_xs: training data points (num_train, dims)
    :param np.ndarray test_xs: test data points (num_test, dims)
    """
    _log_twopi = np.log(2 * np.pi)

    def log_prob_Gaussian(ys, mus, sigmas):
        """
        :param np.ndarray mus: concentration parameters (..., D)
        :param np.ndarray sigmas: concentration parameters (..., D)
        :return:
            log density (...)
        """
        ll = -_log_twopi * 2 * np.log(sigmas + 1e-12) - .5 * (ys - mus) ** 2 / (sigmas**2 + 1e-10)
        return ll.sum(-1)

    def log_prob_Gaussian_mixture(x_eval, ws, mus, sigmas):
        """
        :param np.ndarray x_eval: evaluation locations (..., num_pts, D)
        :param np.ndarray ws: weights (num_pts, num_comps)
        :param np.ndarray mus: weights (num_pts, num_comps, D)
        :param np.ndarray sigmas: weights (num_pts, num_comps, D)
        :return:
            log density (..., num_pts)
        """
        log_ws = np.log(ws)
        ll = log_prob_Gaussian(x_eval[..., None, :], mus, sigmas)  # (..., num_pts, num_comps)
        return logsumexp(log_ws + ll, axis=-1)
    
    # Silverman's rule
    num_train, dims = train_xs.shape
    std_devs = train_xs.std(0, keepdims=True)
    silverman_bws = 0.9 * std_devs * num_train ** (-1 / 5)

    ws = np.ones((1, num_train))
    mus = train_xs[None, ...]
    sigmas = silverman_bws[None, ...]
    return log_prob_Gaussian_mixture(test_xs[:, None, :], ws, mus, sigmas)[:, 0]
    