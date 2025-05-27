
import numpy as np
from scipy.special import logsumexp

def algorithm_func(train_xs, test_xs):
    """
    This algorithm implements Gaussian kernel density estimation using Silverman's rule
    for bandwidth selection and evaluates log likelihood of test data.
    """
    _log_twopi = np.log(2 * np.pi)

    # Gaussian log-density computation
    def log_prob_Gaussian(ys, mus, sigmas):
        """
        Compute log-probabilities of Gaussian kernel.
        """
        ll = -0.5 * (_log_twopi + 2 * np.log(sigmas + 1e-12) + ((ys - mus) ** 2 / (sigmas**2 + 1e-10)))
        return ll.sum(axis=-1)

    # Mixture of Gaussians log-density computation
    def log_prob_Gaussian_mixture(x_eval, ws, mus, sigmas):
        """
        Compute log-probabilities for Gaussian Mixture Model.
        """
        log_ws = np.log(ws)
        ll = log_prob_Gaussian(x_eval[..., None, :], mus, sigmas)  # (..., num_pts, num_comps)
        return logsumexp(log_ws + ll, axis=-1)
    
    # Silverman's heuristic for bandwidth selection
    num_train, dims = train_xs.shape
    std_devs = train_xs.std(axis=0, keepdims=True)
    silverman_bws = 0.9 * std_devs * num_train ** (-1 / 5)

    # Perform kernel density estimation
    ws = np.ones((1, num_train))
    mus = train_xs[None, ...]
    sigmas = silverman_bws[None, ...]
    log_likelihoods = log_prob_Gaussian_mixture(test_xs[:, None, :], ws, mus, sigmas)

    # Return the average log likelihood over test data points
    return log_likelihoods.mean()
