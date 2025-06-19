import numpy as np

def algorithm(train_points):
    """
    This algorithm implements simple binning based on uniform intervals in quantile space.

    :param np.ndarray train_points: training data points (num_train_points,)
    :return:
        bin_edges list[float]
    """
    num_bins = 8  # an arbitrarily chosen number of bins

    quantile_vals = np.linspace(0, 1, num_bins + 1)
    sort_inds = np.argsort(train_points)

    # get indices corresponding to nearest to quantile values
    quantile_inds = [int(v * (len(sort_inds) - 1)) for v in quantile_vals]

    # convert from quantile to actual values
    bin_edges = np.sort(train_points)[quantile_inds]
    bin_edges = [-1] + bin_edges.tolist()[1:-1] + [1]  # ensure edges cover the valid domain

    return bin_edges