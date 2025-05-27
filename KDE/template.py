import numpy as np
import time

import src


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
algorithm_prompt = """
Write a function that takes in training data points of shape (num_pts, dims), as well as test data (num_test, dims), and spits out a scalar which is the estimated log likelihood of the test data.
You are implementing kernel density estimation using some form of bandwidth selection heuristic for your base kernel of choice.
Use only NumPy and SciPy and implement the rest from scratch.
"""


first_algorithm = """
import numpy as np
from scipy.special import logsumexp


def algorithm_func(train_xs, test_xs):
    \"\"\"
    This algorithm implements Gaussian kernel density estimation using Silverman

    :param np.ndarray train_xs: training data points (num_train, dims)
    :param np.ndarray test_xs: test data points (num_test, dims)
    \"\"\"
    _log_twopi = np.log(2 * np.pi)

    def log_prob_Gaussian(ys, mus, sigmas):
        \"\"\"
        :param np.ndarray mus: concentration parameters (..., D)
        :param np.ndarray sigmas: concentration parameters (..., D)
        :return:
            log density (...)
        \"\"\"
        ll = -_log_twopi * 2 * np.log(sigmas + 1e-12) - .5 * (ys - mus) ** 2 / (sigmas**2 + 1e-10)
        return ll.sum(-1)

    def log_prob_Gaussian_mixture(x_eval, ws, mus, sigmas):
        \"\"\"
        :param np.ndarray x_eval: evaluation locations (..., num_pts, D)
        :param np.ndarray ws: weights (num_pts, num_comps)
        :param np.ndarray mus: weights (num_pts, num_comps, D)
        :param np.ndarray sigmas: weights (num_pts, num_comps, D)
        :return:
            log density (..., num_pts)
        \"\"\"
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
    """



# LLM feedback
feedback_prompt = """
Target score to maximize is your test likelihoods on a fixed set of problem instances (different datasets, here we test on 64 instances), we provide both per instance likelihoods and a single average over instances. In addition, you will get:
- Train data likelihoods (per instance and averaged)
- Time to run evaluation for all instances (we want this to be as low as possible, but prioritize the target score)
- Code complexity score (length of the Python code in bytes, we prefer algorithms where this is not too high)
You can come up with your own internal objective function (e.g. average test likelihood penalised by code complexity or so)
"""


def EvaluateAlgorithm(instances_seed, algo_script_file: str):
    """
    Generate problem instances and applies it to the algorithm

    :return:
        evaluation performance results that is added to feedback prompt
    """
    with open(algo_script_file, 'r') as f:
        algo_code_len = len(f.read())

    algorithm_func, = src.utils.import_model_specific_symbols(algo_script_file, ['algorithm_func'])
    
    # evaluation
    num_instances = 64
    start_time = time.time()

    test_lls, train_lls = [], []
    for i in range(num_instances):
        seed = instances_seed + i

        rng_samps = np.random.default_rng(seed + 1)

        num_samples = 2 ** 8
        z = rng_samps.normal(size=(num_samples, 2))
        rng_params = np.random.default_rng(seed)
        train_xs = generate_samples(rng_params, z)

        num_samples = 2 ** 5
        z = rng_samps.normal(size=(num_samples, 2))
        rng_params = np.random.default_rng(seed)
        test_xs = generate_samples(rng_params, z)

        test_ll_xs = algorithm_func(train_xs, test_xs)
        train_ll_xs = algorithm_func(train_xs, train_xs)
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


def ConstructFeedback(evaluation_results):
    """
    Construct the performance feedback prompt
    """
    results_str = str(evaluation_results)
    
    return results_str