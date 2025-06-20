import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time


# problem instance
def generate_samples(x, width=8, layers=4, sigma=3.):
    """
    Transform input samples through a small Siren MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = np.random.normal(size=(x.shape[1], width)) * sigma / np.sqrt(x.shape[1] + width)
        b = np.random.normal(size=(width,)) * sigma / np.sqrt(width)
        x = np.sin(x @ W + b)

    W_out = np.random.normal(size=(x.shape[1], dims)) * sigma / np.sqrt(x.shape[1] + dims)
    b_out = np.random.normal(size=(dims)) * sigma / np.sqrt(dims)
    x = np.sin(x @ W_out + b_out)

    return x


def histogram_log_likelihood(x_eval, bin_edges, x_train, upper_jitter=1e-6):
    """
    Evaluate a histogram density defined by bin edges and occupancies
    """

    bin_widths = np.diff(bin_edges)
    bin_edges_ = bin_edges[:-1] + [bin_edges[-1] + upper_jitter]

    # histogram
    bin_indices = np.digitize(x_train, bins=bin_edges)
    bin_occup = np.bincount(bin_indices)[1:]
    bin_occup = np.maximum(bin_occup, 1e-5)  # avoid 0 for -inf test likelihood
    probs = bin_occup / np.sum(bin_occup)

    # evaluate
    bin_indices = np.digitize(x_eval, bins=bin_edges_) - 1
    log_probs = np.log(probs[bin_indices] / bin_widths[bin_indices])

    return log_probs


def generate_instance(num_train_points: int, num_test_points: int, visualise: bool = False) -> dict:
    """
    Generate problem instance
    """
    if visualise:
        z = np.random.normal(size=((num_train_points + num_test_points) * 1024, 1))
    else:
        z = np.random.normal(size=(num_train_points + num_test_points, 1))
    points = generate_samples(z)

    train_points = points[:num_train_points, 0]
    test_points = points[num_train_points:num_train_points + num_test_points, 0]

    return dict(
        train_points=train_points,
        test_points=test_points,
        points=points[:, 0],
    )


def algorithm(train_points):
    """
    Template for the algorithm to be implemented

    :param np.ndarray train_points: training data points (num_train_points,)
    :return:
        bin_edges list[float]
    """
    raise NotImplementedError


def evaluate_algorithm(instance: dict, algorithm: callable, plot_image_path: str = None) -> dict:
    bin_edges = algorithm(instance["train_points"])
    test_ll_xs = histogram_log_likelihood(instance["test_points"], bin_edges, instance["train_points"])
    train_ll_xs = histogram_log_likelihood(instance["test_points"], bin_edges, instance["train_points"])

    if plot_image_path:
        hist, _ = np.histogram(instance["points"], bins=bin_edges, density=True)
        density_bar = hist[np.newaxis, :]  # shape (1, bins)
        
        hist_gt, bin_edges_gt = np.histogram(instance["points"], bins=100, density=True)
        density_bar_gt = hist_gt[np.newaxis, :]  # shape (1, bins)
        
        plt.figure(figsize=(10, 2.5))
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(density_bar, aspect='auto', cmap='Blues',
                extent=[bin_edges[0], bin_edges[-1], 0, 1])
        ax1.set_yticks([])
        ax1.set_title(f"Estimated Density (log likelihood: {test_ll_xs.mean().item():.2f})")

        ax2 = plt.subplot(2, 1, 2)
        ax2.imshow(density_bar_gt, aspect='auto', cmap='Oranges',
                extent=[bin_edges_gt[0], bin_edges_gt[-1], 0, 1])
        ax2.set_yticks([])
        ax2.set_title("Ground Truth Density")

        # Save to file
        plt.tight_layout()
        plt.savefig(plot_image_path)
        plt.close()

    return dict(
        train_log_likelihood=train_ll_xs.mean().item(),
        test_log_likelihood=test_ll_xs.mean().item()
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
    parser.add_argument("--output", type=str, help="Output directory for visualisations", default="output")
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
        generate_instance(**parameters, visualise=(i < args.visualisations)) 
        for i in range(args.instances)
    ]
    results = [
        evaluate_algorithm(instance, module.algorithm, None if i >= args.visualisations else f"{args.output}/visualisation_{i}.png")
        for i, instance in enumerate(instances)
    ]
    elapsed_time = time() - start
    train_log_likelihoods = [result["train_log_likelihood"] for result in results]
    test_log_likelihoods = [result["test_log_likelihood"] for result in results]
    
    results = {
        "train_log_likelihoods": train_log_likelihoods,
        "test_log_likelihoods": test_log_likelihoods,
        "train_log_likelihood_average": np.mean(train_log_likelihoods).item(),
        "test_log_likelihood_average": np.mean(test_log_likelihoods).item(),
        "elapsed_time": elapsed_time,
    }
    if args.evaluation:
        with open(args.evaluation, "w") as f:
            json.dump(results, f)
    else:
        print(json.dumps(results, indent=2))
