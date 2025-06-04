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
    Write a function that takes in data points of shape (num_train_pts, dims) and returns for each point the index of the cluster it belongs to [1, ... K] for some given number of clusters K > 0.
    Use only NumPy and implement the rest from scratch.
""")



# LLM feedback
feedback_prompt = textwrap.dedent("""
    Target score to maximize is a graph-based connectivity score of all clusters on a fixed set of problem instances (different datasets, here we test on 64 instances).
""")



###
# WIP #
###