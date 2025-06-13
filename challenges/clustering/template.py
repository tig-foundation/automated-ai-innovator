import numpy as np
from sklearn.neighbors import NearestNeighbors

import time
import textwrap

import autoinnovator


# problem instance
def generate_samples(rng, x, width=12, layers=5):
    """
    Transform input samples through a small MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = rng.normal(size=(x.shape[1], width)) / np.sqrt(x.shape[1])
        b = 0.01 * rng.normal(size=(width,)) + 0.1 * rng.integers(-5, 6, size=(1000,)).max()
        x = np.maximum(x @ W + b, 0)

    W_out = rng.normal(size=(x.shape[1], dims))
    b_out = rng.normal(size=(dims))
    x = x @ W_out + b_out

    return x


# algorithm
algorithm_prompt = textwrap.dedent("""
    Write a function that takes in data points of shape (num_train_pts, dims) and returns for each point the index of the cluster it belongs to [1, ... K] for some given number of clusters K > 0.
    Use only NumPy and implement the rest from scratch.
""")


first_algorithm = textwrap.dedent("""
    import numpy as np

    def algorithm_func(train_xs):
        \"\"\"
        This algorithm implements simple binning based on uniform intervals in quantile space.

        :param np.ndarray train_xs: training data points (num_train_pts, dims)
        :return:
            bin_edges list[float]
        \"\"\"
        num_bins = 8  # an arbitrarily chosen number of bins

        quantile_vals = np.linspace(0, 1, num_bins + 1)
        sort_inds = np.argsort(x_vals)

        # get indices corresponding to nearest to quantile values
        quantile_inds = [int(v * (len(sort_inds) - 1)) for v in quantile_vals]

        # convert from quantile to actual values
        bin_edges = np.sort(x_vals)[quantile_inds]
        bin_edges = [-1] + bin_edges.tolist()[1:-1] + [1]  # ensure edges cover the valid domain

        return bin_edges
""")



# LLM feedback
feedback_prompt = textwrap.dedent("""
    Target score to maximize is a graph-based connectivity score of all clusters on a fixed set of problem instances (different datasets, here we test on 64 instances).
""")



def pairwise_connectivity_score(X, labels, n_neighbors=10):
    """
    Compute the connectivity score for clustering.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), the data
    - labels: array-like of shape (n_samples,), cluster labels
    - n_neighbors: int, number of nearest neighbors to consider

    Returns:
    - connectivity_score: float (lower is better)
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    n_samples = X.shape[0]
    
    # Build nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Exclude self-neighbors (first column is self)
    indices = indices[:, 1:]

    score = 0.0
    for i in range(n_samples):
        for rank, j in enumerate(indices[i]):
            if labels[i] != labels[j]:
                # Penalize by inverse rank (rank+1 since rank starts at 0)
                score += 1.0 / (rank + 1)
    
    return score


def EvaluateAlgorithm(instances_seed: int, challenge_params: dict, algo_script_file: str):
    """
    Generate problem instances and applies it to the algorithm

    :return:
        evaluation performance results that is added to feedback prompt
    """
    dims = 1

    with open(algo_script_file, 'r') as f:
        algo_code_len = len(f.read())

    algorithm_func, = autoinnovator.utils.import_model_specific_symbols(algo_script_file, ['algorithm_func'])
    
    # evaluation
    num_instances = 64
    start_time = time.time()

    test_lls, train_lls = [], []
    for i in range(num_instances):
        seed = instances_seed + i

        rng_samps = np.random.default_rng(seed + 1)

        num_samples = challenge_params['num_train_pts']
        z = rng_samps.normal(size=(num_samples, dims))
        rng_params = np.random.default_rng(seed)
        train_xs = generate_samples(rng_params, z)[:, 0]

        num_samples = challenge_params['num_test_pts']
        z = rng_samps.normal(size=(num_samples, dims))
        rng_params = np.random.default_rng(seed)
        test_xs = generate_samples(rng_params, z)[:, 0]

        bin_edges = algorithm_func(train_xs)
        test_ll_xs = histogram_log_likelihood(test_xs, bin_edges, train_xs)
        train_ll_xs =histogram_log_likelihood(test_xs, bin_edges, train_xs)

        test_lls.append(test_ll_xs.mean().item())
        train_lls.append(train_ll_xs.mean().item())

    elapsed_time_seconds = time.time() - start_time
    results = {
        'test_log_likelihood_average': np.array(test_lls).mean().item(), 
        'test_log_likelihood_per_instance': test_lls, 
        'train_log_likelihood_average': np.array(train_lls).mean().item(), 
        'train_log_likelihood_per_instance': train_lls, 
        'algo_code_len': algo_code_len, 
        'elapsed_time_seconds': elapsed_time_seconds, 
    }
    return results


def ConstructFeedback(evaluation_results: dict):
    """
    Construct the performance feedback prompt
    """
    results_str = str(evaluation_results)
    
    return results_str