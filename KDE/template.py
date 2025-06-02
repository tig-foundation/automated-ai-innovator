import numpy as np
from scipy.special import logsumexp, log_softmax

import time
import textwrap

import autoinnovator


# problem instance
def generate_samples(rng, x, width=8, layers=3):
    """
    Transform input samples through a small MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = rng.normal(size=(x.shape[1], width)) / np.sqrt(x.shape[1])
        b = rng.normal(size=(width,))
        x = np.tanh(x @ W + b)

    W_out = rng.normal(size=(x.shape[1], dims))
    b_out = rng.normal(size=(dims))
    x = x @ W_out + b_out

    return x


# algorithm
algorithm_prompt = textwrap.dedent("""
    Write a function that takes in training data points of shape (num_train_pts, dims) and returns Gaussian mixture model parameters: component unnormalised weight logits of shape (num_comps,), component means (num_comps, dims) and component covariances (num_comps, dims, dims). 
    You are implementing kernel density estimation using some form of bandwidth selection heuristic for your base kernel of choice.
    Use only NumPy and implement the rest from scratch.
""")


first_algorithm = textwrap.dedent("""
    import numpy as np


    def algorithm_func(train_xs):
        \"\"\"
        This algorithm implements Gaussian kernel density estimation using Silverman

        :param np.ndarray train_xs: training data points (num_train_pts, dims)
        :return:
            weight_logits (num_comps,)
            mus (num_comps, dims)
            covs (num_comps, dims, dims)
        \"\"\"
        num_pts, dims = train_xs.shape
        std_devs = train_xs.std(0, keepdims=True)
        silverman_bws = 1.06 * std_devs * num_pts ** (-1 / 5)  # Silverman's rule

        num_comps = num_pts  # place one Gaussian at every point
        weight_logits = np.zeros(num_comps)  # uniform weighting
        mus = train_xs
        sigmas = np.broadcast_to(silverman_bws, (num_comps, dims))
        covs = np.einsum('...i,jk->...jk', sigmas ** 2, np.eye(dims))

        return weight_logits, mus, covs
""")



# LLM feedback
feedback_prompt = textwrap.dedent("""
    Target score to maximize is test likelihoods (Gaussian mixture model evaluated on num_test_pts test data points) on a fixed set of problem instances (different datasets, here we test on 64 instances), we provide both per instance likelihoods and a single average over instances. In addition, you will get:
    - Train data likelihoods (per instance and averaged)
    - Time to run evaluation for all instances (we want this to be as low as possible, but prioritize the target score)
    - Code complexity score (length of the Python code in bytes, we prefer algorithms where this is not too high)
    You can come up with your own internal objective function (e.g. average test likelihood penalised by code complexity or so)
""")


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

    test_lls, train_lls = [], []
    for i in range(num_instances):
        seed = instances_seed + i

        rng_samps = np.random.default_rng(seed + 1)

        num_samples = challenge_params['num_train_pts']
        z = rng_samps.normal(size=(num_samples, dims))
        rng_params = np.random.default_rng(seed)
        train_xs = generate_samples(rng_params, z)

        num_samples = challenge_params['num_test_pts']
        z = rng_samps.normal(size=(num_samples, dims))
        rng_params = np.random.default_rng(seed)
        test_xs = generate_samples(rng_params, z)

        weight_logits, mus, covs = algorithm_func(train_xs)
        ws_logits_batch, mus_batch, covs_batch = weight_logits[None], mus[None], covs[None]
        test_ll_xs = log_prob_multivariate_Gaussian_mixture(test_xs, ws_logits_batch, mus_batch, covs_batch)
        train_ll_xs = log_prob_multivariate_Gaussian_mixture(train_xs, ws_logits_batch, mus_batch, covs_batch)

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