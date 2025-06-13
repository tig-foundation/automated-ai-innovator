import numpy as np

import time
import textwrap

import autoinnovator


# problem instance
def generate_samples(rng, x, width=8, layers=4, sigma=3.):
    """
    Transform input samples through a small Siren MLP
    
    :param np.ndarray x: input samples (num_pts, dims)
    """
    dims = x.shape[1]

    for _ in range(layers):
        W = rng.normal(size=(x.shape[1], width)) * sigma / np.sqrt(x.shape[1] + width)
        b = rng.normal(size=(width,)) * sigma / np.sqrt(width)
        x = np.sin(x @ W + b)

    W_out = rng.normal(size=(x.shape[1], dims)) * sigma / np.sqrt(x.shape[1] + dims)
    b_out = rng.normal(size=(dims)) * sigma / np.sqrt(dims)
    x = np.sin(x @ W_out + b_out)

    return x



# algorithm
algorithm_prompt = textwrap.dedent("""
    Write a function that takes in training data points of shape (num_train_pts,) in the 1D domain [-1, 1] and returns bin edge values that cover the entire domain which we use to bin the data points into a histogram density.
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



# first_algorithm = textwrap.dedent("""
#     import numpy as np

#     def algorithm_func(train_xs):
#         \"\"\"
#         This algorithm implements binning with a greedy decision tree based on max log likelihood splits.

#         :param np.ndarray train_xs: training data points (num_train_pts, dims)
#         :return:
#             bin_edges list[float]
#         \"\"\"
#         min_bin_width = 1e-3
#         num_bins = 8  # an arbitrarily chosen number of bins

#         def compute_log_likelihood(x, bins, counts):
#             bin_indices = np.digitize(x, bins) - 1
#             bin_widths = np.diff(bins)
#             probs = counts / np.sum(counts)
            
#             mask = (bin_indices >= 0) & (bin_indices < len(probs))
#             log_probs = np.log(probs[bin_indices[mask]] / bin_widths[bin_indices[mask]])
#             return np.sum(log_probs)

#         x_vals = np.sort(x_vals)
#         intervals = [(x_vals[0], x_vals[-1])]
#         bins = [x_vals[0], x_vals[-1]]
        
#         for _ in range(num_bins - 1):
#             best_gain = -np.inf
#             best_split = None
#             best_new_bins = None
            
#             for i, (a, b) in enumerate(intervals):
#                 candidates = x_vals[(x_vals > a) & (x_vals < b)]
#                 for s in candidates:
#                     if s - a < min_bin_width or b - s < min_bin_width:
#                         continue
#                     test_bins = bins[:i+1] + [s] + bins[i+1:]
#                     test_counts, _ = np.histogram(x_vals, bins=test_bins)
#                     ll = compute_log_likelihood(x_vals, test_bins, test_counts)
#                     if ll > best_gain:
#                         best_gain = ll
#                         best_split = (i, s)
#                         best_new_bins = test_bins
            
#             if best_split is None:
#                 break
#             bins = best_new_bins
#             a, b = intervals.pop(best_split[0])
#             s = best_split[1]
#             intervals.insert(best_split[0], (s, b))
#             intervals.insert(best_split[0], (a, s))

#         return np.array(bins)
# """)



# LLM feedback
feedback_prompt = textwrap.dedent("""
    Target score to maximize is test likelihoods (histogram density evaluated on num_test_pts test data points) on a fixed set of problem instances (different datasets, here we test on 64 instances), we provide both per instance likelihoods and a single average over instances. In addition, you will get:
    - Train data likelihoods (per instance and averaged)
    - Time to run evaluation for all instances (we want this to be as low as possible, but prioritize the target score)
    - Code complexity score (length of the Python code in bytes, we prefer algorithms where this is not too high)
    You can come up with your own internal objective function (e.g. average test likelihood penalised by code complexity or so)
""")



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
        train_ll_xs = histogram_log_likelihood(test_xs, bin_edges, train_xs)

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