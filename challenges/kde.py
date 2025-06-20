import importlib
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, log_softmax
from time import time


# problem instance
def generate_samples(x, width=8, layers=3):
    """
    Transform input samples through a small MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = np.random.normal(size=(x.shape[1], width)) / np.sqrt(x.shape[1])
        b = np.random.normal(size=(width,))
        x = np.tanh(x @ W + b)

    W_out = np.random.normal(size=(x.shape[1], dims))
    b_out = np.random.normal(size=(dims))
    x = x @ W_out + b_out

    return x


_log_twopi = np.log(2 * np.pi)


def log_prob_multivariate_Gaussian(xs, mus, covs):
    """
    :param np.ndarray xs: evaluation points (..., dims)
    :param np.ndarray mus: means (..., dims)
    :param np.ndarray covs: covariance matrices (..., dims, dims)
    :return:
        log probabilities (...)
    """
    D = xs.shape[-1]
    delta = xs - mus  # (..., D)
    chol = np.linalg.cholesky(covs)  # (..., D, D)
    # Solve instead of inverse for numerical stability
    delta_sol = np.linalg.solve(chol, delta[..., None])  # (..., D, 1)
    mahal = np.sum(delta_sol.squeeze(-1) ** 2, axis=-1)  # (...)
    log_det = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    return -0.5 * (D * _log_twopi + log_det + mahal)


def log_prob_multivariate_Gaussian_mixture(xs, ws_logits, mus, covs):
    """
    :param np.ndarray xs: evaluation points (..., num_pts, dims)
    :param np.ndarray ws: mixture weights (..., num_pts, num_comps)
    :param np.ndarray mus: means (..., num_pts, num_comps, dims)
    :param np.ndarray covs: covariances (..., num_pts, num_comps, dims, dims)
    :return:
        log probabilities (..., num_pts)
    """
    log_ws = log_softmax(ws_logits, axis=-1)  # (..., num_pts, num_comps)
    xs_exp = xs[..., :, None, :]  # (..., num_pts, 1, D)
    log_probs = log_prob_multivariate_Gaussian(xs_exp, mus, covs)  # (..., num_pts, num_comps)
    return logsumexp(log_ws + log_probs, axis=-1)  # (..., num_pts)


def generate_instance(dims: int, num_train_points: int, num_test_points: int, visualise: bool = False) -> dict:
    """
    Generate problem instance
    """
    if visualise:
        z = np.random.normal(size=((num_train_points + num_test_points) * 1024, dims))
    else:
        z = np.random.normal(size=(num_train_points + num_test_points, dims))
    points = generate_samples(z)

    train_points = points[:num_train_points]
    test_points = points[num_train_points:num_train_points + num_test_points]

    return dict(
        train_points=train_points,
        test_points=test_points,
        points=points,
    )


def algorithm(train_points) -> tuple:
    """
    Template for the algorithm to be implemented

    :param np.ndarray train_points: training data points (num_train_pts, dims)
    :return:
        weight_logits (num_comps,)
        mus (num_comps, dims)
        covs (num_comps, dims, dims)
    """
    raise NotImplementedError


def evaluate_algorithm(instance: dict, algorithm: callable, plot_image_path: str = None) -> dict:
    weight_logits, mus, covs = algorithm(instance["train_points"])
    ws_logits_batch, mus_batch, covs_batch = weight_logits[None], mus[None], covs[None]
    test_ll_xs = log_prob_multivariate_Gaussian_mixture(instance["test_points"], ws_logits_batch, mus_batch, covs_batch)
    train_ll_xs = log_prob_multivariate_Gaussian_mixture(instance["train_points"], ws_logits_batch, mus_batch, covs_batch)

    if plot_image_path:
        grid_pts = 128
        x = np.linspace(instance["points"][:, 0].min() - 0.5, instance["points"][:, 0].max() + 0.5, grid_pts)
        y = np.linspace(instance["points"][:, 1].min() - 0.5, instance["points"][:, 1].max() + 0.5, grid_pts)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X, Y], axis=-1).reshape(-1, 2)  # (num_pts, 2)
        log_prob_xy = log_prob_multivariate_Gaussian_mixture(xy, ws_logits_batch, mus_batch, covs_batch).reshape(grid_pts, grid_pts)
        P = np.exp(log_prob_xy)
        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.contourf(X, Y, P, vmax=P.max() * 1.25, levels=50, cmap="Blues")
        ax1.set_title(f"Estimate Density (log likelihood: {train_ll_xs.mean().item():.2f})")
        ax1.grid(True)
        ax2 = plt.subplot(1, 2, 2)
        ax2.hist2d(instance["points"][:, 0], instance["points"][:, 1], bins=100, density=True, cmap='Oranges')
        ax2.set_title("Empirical Sampled Density")
        ax2.grid(True)
        ax1.set_xlim(ax2.get_xlim())
        ax1.set_ylim(ax2.get_ylim())
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
        generate_instance(**parameters, visualise=(i < args.visualisations)) 
        for i in range(args.instances)
    ]
    results = [
        evaluate_algorithm(instance, module.algorithm, None if i >= args.visualisations else args.output.format(i=i))
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
        "elapsed_seconds": elapsed_time,
    }
    if args.evaluation:
        with open(args.evaluation, "w") as f:
            json.dump(results, f)
    else:
        print(json.dumps(results, indent=2))
