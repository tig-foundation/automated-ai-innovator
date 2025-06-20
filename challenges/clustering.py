import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.neighbors import NearestNeighbors


# problem instance
def generate_samples(x, width=12, layers=5):
    """
    Transform input samples through a small MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = np.random.normal(size=(x.shape[1], width)) / np.sqrt(x.shape[1])
        b = 0.01 * np.random.normal(size=(width,)) + 0.1 * np.random.randint(-5, 6, size=(1000,)).max()
        x = np.maximum(x @ W + b, 0)

    W_out = np.random.normal(size=(x.shape[1], dims))
    b_out = np.random.normal(size=(dims))
    x = x @ W_out + b_out

    return x


def pairwise_connectivity_score(X, labels, n_neighbors=10):
    """
    Compute the connectivity score for clustering.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), the data
    - labels: array-like of shape (n_samples,), cluster labels
    - n_neighbors: int, number of nearest neighbors to consider

    Returns:
    - connectivity_score: float (value is <= 0, higher is better)
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
                score -= 1.0 / (rank + 1)
    
    return score


def generate_instance(dims: int, num_points: int) -> dict:
    """
    Generate problem instance
    """
    z = np.random.normal(size=(num_points, dims))
    points = generate_samples(z)

    return dict(
        points=points,
    )


def algorithm(points):
    """
    Template for the algorithm to be implemented

    :param np.ndarray points with shape (num_points, dims)
    :return:
        cluster indices (num_points,)
    """
    raise NotImplementedError


def evaluate_algorithm(instance: dict, algorithm: callable, plot_image_path: str = None) -> dict:
    cluster_inds = algorithm(instance["points"])
    cluster_score = pairwise_connectivity_score(instance["points"], cluster_inds, n_neighbors=10)

    if plot_image_path:
        plt.figure()
        plt.scatter(instance["points"][:, 0], instance["points"][:, 1], c=cluster_inds, cmap="Set1", s=5, alpha=0.25)
        plt.title(f"Clustering (score: {cluster_score:.2f})")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(plot_image_path)
        plt.close()

    return dict(
        cluster_score=cluster_score
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KDE Challenge")
    parser.add_argument("algorithm", type=str, help="Path to the algorithm python file")
    parser.add_argument("parameters", type=str, help="JSON string of parameters")
    parser.add_argument("seed", type=int, help="Seed for random number generation")
    parser.add_argument("instances", type=int, help="Number of instances to generate")
    parser.add_argument("--evaluation", type=str, help="File path to save JSON evaluation results. Prints to stdout by default", default=None)
    parser.add_argument("--output", type=str, help="Path format for visualisations. Default {i:03}.png", default="{i:03}.png")
    parser.add_argument("--visualisations", type=int, help="Number of instances to visualise", default=0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.evaluation and (evaluation_dir := os.path.dirname(args.evaluation)):
        os.makedirs(evaluation_dir, exist_ok=True)

    parameters = json.loads(args.parameters)
    spec = importlib.util.spec_from_file_location(
        os.path.splitext(os.path.basename(args.algorithm))[0], 
        args.algorithm
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    start = time()
    np.random.seed(args.seed)
    instances = [
        generate_instance(**parameters) 
        for i in range(args.instances)
    ]
    results = [
        evaluate_algorithm(instance, module.algorithm, None if i >= args.visualisations else args.output.format(i=i))
        for i, instance in enumerate(instances)
    ]
    elapsed_time = time() - start
    cluster_scores = [result["cluster_score"] for result in results]
    
    results = {
        "cluster_scores_average": np.array(cluster_scores).mean().item(), 
        "cluster_scores_per_instance": cluster_scores,
        "elapsed_seconds": elapsed_time, 
    }
    if args.evaluation:
        with open(args.evaluation, "w") as f:
            json.dump(results, f)
    else:
        print(json.dumps(results, indent=2))
