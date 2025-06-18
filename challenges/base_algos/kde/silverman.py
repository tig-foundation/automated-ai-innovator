import numpy as np


def algorithm(train_points):
    """
    This algorithm implements Gaussian kernel density estimation using Silverman

    :param np.ndarray train_points: training data points (num_train_pts, dims)
    :return:
        weight_logits (num_comps,)
        mus (num_comps, dims)
        covs (num_comps, dims, dims)
    """
    num_pts, dims = train_points.shape
    std_devs = train_points.std(0, keepdims=True)
    silverman_bws = 1.06 * std_devs * num_pts ** (-1 / 5)  # Silverman's rule

    num_comps = num_pts  # place one Gaussian at every point
    weight_logits = np.zeros(num_comps)  # uniform weighting
    mus = train_points
    sigmas = np.broadcast_to(silverman_bws, (num_comps, dims))
    covs = np.einsum('...i,jk->...jk', sigmas ** 2, np.eye(dims))

    return weight_logits, mus, covs