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
    from sklearn.cluster import KMeans

    def algorithm_func(xs):
        \"\"\"
        This algorithm implements simple k-means clustering with a fixed arbitrary number of clusters (not data-based).

        :param np.ndarray xs: training data points (num_pts, dims)
        :return:
            cluster indices (num_pts,)
        \"\"\"
        n_clusters = 3  # arbitrarily chosen
        cluster_inds = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(xs) + 1

        return cluster_inds
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
    dims = challenge_params['dims']

    with open(algo_script_file, 'r') as f:
        algo_code_len = len(f.read())

    algorithm_func, = autoinnovator.utils.import_model_specific_symbols(algo_script_file, ['algorithm_func'])
    
    # evaluation
    num_instances = 64
    start_time = time.time()

    cluster_scores = []
    for i in range(num_instances):
        seed = instances_seed + i

        rng_samps = np.random.default_rng(seed + 1)

        num_samples = challenge_params['num_pts']
        z = rng_samps.normal(size=(num_samples, dims))
        rng_params = np.random.default_rng(seed)
        xs = generate_samples(rng_params, z)

        cluster_inds = algorithm_func(xs)
        score = pairwise_connectivity_score(xs, cluster_inds, n_neighbors=10)

        cluster_scores.append(score)

    elapsed_time_seconds = time.time() - start_time
    results = {
        'cluster_scores_average': np.array(cluster_scores).mean().item(), 
        'cluster_scores_per_instance': cluster_scores, 
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