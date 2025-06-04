import numpy as np

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
    Write a function that takes in training data points of shape (num_train_pts,) in the 1D domain [-1, 1] and returns bin edge values that cover the entire domain which we use to bin the data points into a histogram density.
    Use only NumPy and implement the rest from scratch.
""")



# LLM feedback
feedback_prompt = textwrap.dedent("""
    Target score to maximize is test likelihoods (histogram density evaluated on num_test_pts test data points) on a fixed set of problem instances (different datasets, here we test on 64 instances), we provide both per instance likelihoods and a single average over instances. In addition, you will get:
    - Train data likelihoods (per instance and averaged)
    - Time to run evaluation for all instances (we want this to be as low as possible, but prioritize the target score)
    - Code complexity score (length of the Python code in bytes, we prefer algorithms where this is not too high)
    You can come up with your own internal objective function (e.g. average test likelihood penalised by code complexity or so)
""")



###
# WIP #
###