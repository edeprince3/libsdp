import time
import jax
import jax.numpy as jnp
from jax import tree_util, flatten_util

from functools import partial # Import partial

class Timer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self, verbose=True):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        if self.verbose:
            print(f"Elapsed time: {elapsed_time:.4f} seconds")

from libsdp.jax_utilities import tree_real_matrix_inner_product, ravel_dict_ordered