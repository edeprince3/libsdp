import jax
import jax.numpy as jnp
from jax import flatten_util

def get_pytree_memory_gb(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    # Sum bytes of all leaves
    total_bytes = sum(leaf.size * leaf.itemsize for leaf in leaves)
    # Convert to GiB (1024^3 bytes)
    return total_bytes / (1024**3)

@jax.jit
def tree_real_matrix_inner_product(params1, params2):
    return jax.tree_util.tree_map(lambda m1, m2: m1 @ m2.T, params1, params2)

def ravel_dict_ordered(pytree, ordered_keys):
    """
    Ravels a dictionary of arrays into a 1D array based on a specific key order.
    
    Args:
        pytree: Dict[str, Array] (or nested pytrees)
        ordered_keys: List[str] specifying the order of keys to flatten.
        
    Returns:
        flat_array: 1D JAX array.
        unravel_fn: Function(flat_array) -> Dict[str, Array]
    """
    # 1. Extract the values in the exact order requested.
    #    This avoids complex path matching and string parsing errors.
    values_in_order = [pytree[key] for key in ordered_keys]

    # 2. Use JAX to ravel this specific list of values.
    #    JAX handles the shapes/sizes of the arrays (or sub-trees) automatically.
    flat_array, unravel_list_fn = flatten_util.ravel_pytree(values_in_order)

    # 3. Create a wrapper to convert the unraveled list back into a dict.
    def unravel_fn(flat_vec):
        # Recover the list of values
        recovered_values = unravel_list_fn(flat_vec)
        # Zip it back with the keys to reform the dictionary
        return dict(zip(ordered_keys, recovered_values))

    return flat_array, unravel_fn
