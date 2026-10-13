import jax
import jax.numpy as jnp
from jax.experimental import sparse
import jax.flatten_util
import numpy as np
import h5py
from libsdp.sdpa_file_io import read_sdpa_problem
from libsdp.jax_utilities import ravel_dict_ordered

def generate_problem_from_sdpa(filename, return_matrices=False):
    """
    Parse an SDPA sparse file and return AX, adjoint, and CX operators for ADMM.

    Operators use dictionaries of matrix blocks keyed by ``block_names``.
    Each constraint row and its right-hand side are divided by the L2 norm
    of its stored coefficients (the Frobenius norm when entries are unique).
    Rows with norms below 1e-15 remain unscaled. The objective C is unchanged.

    Args:
        filename (str): Path to the .dat-s file.
        return_matrices (bool): Append the sparse A matrix and C vector.

    Returns:
        tuple: ``(AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X)``.
            AX_fn maps a block dictionary X to ``{'y': A @ flatten(X)}``.
            ATy_fn maps ``{'y': y}`` to the block dictionary for A.T @ y.
            CX_fn maps X to the scalar objective using the unchanged C.
            b is the normalized constraint vector; shapes gives block shapes.
            block_names contains string indices in block order.
            flatten_X maps X to a flat array and its inverse function.
            If return_matrices is True, normalized A_mat and unchanged C_vec
            (JAX BCOO arrays) are appended, giving nine return values.
    """
    # 1. Read SDPA problem
    b_list, Fi, block_dims = read_sdpa_problem(filename)
    block_names = [str(xx) for xx in range(len(block_dims))] 
    return generate_problem_from_data(b_list, Fi, block_dims, block_names=block_names, return_matrices=return_matrices)


def generate_problem_from_data(b_list, Fi, block_dims, block_names=None, return_matrices=False):
    """
    Generate AX, adjoint, and CX operators for ADMM from parsed SDPA data.

    Operators use dictionaries of matrix blocks keyed by ``block_names``.
    Each constraint row and its right-hand side are divided by the L2 norm
    of its stored coefficients (the Frobenius norm when entries are unique).
    Rows with norms below 1e-15 remain unscaled. The objective C is unchanged.

    Args:
        b_list (list): Constraint vector elements.
        Fi (list): Sparse matrices; F0 is the objective, F1...Fm are constraints.
        block_dims (list): List of block dimensions.
        block_names (list, optional): Dictionary keys in block order; defaults
            to string indices.
        return_matrices (bool): Append the sparse A matrix and C vector.

    Returns:
        tuple: ``(AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X)``.
            AX_fn maps a block dictionary X to ``{'y': A @ flatten(X)}``.
            ATy_fn maps ``{'y': y}`` to the block dictionary for A.T @ y.
            CX_fn maps X to the scalar objective using the unchanged C.
            b is the normalized constraint vector; shapes gives block shapes.
            block_names contains the dictionary keys in block order.
            flatten_X maps X to a flat array and its inverse function.
            If return_matrices is True, normalized A_mat and unchanged C_vec
            (JAX BCOO arrays) are appended, giving nine return values.
    """
    b = jnp.array(b_list)
    m = len(b)
    
    # 2. Block Indexing & Offsets
    # We flatten X = [Block1, Block2, ...] into one long vector.
    
    block_sizes = [abs(d)**2 for d in block_dims]
    block_offsets = [0]
    for size in block_sizes:
        block_offsets.append(block_offsets[-1] + size)
    
    total_dim = block_offsets[-1]
    
    # Shapes for reconstruction/initializing X
    shapes = [(abs(d), abs(d)) for d in block_dims]

    if block_names is None:
        block_names = [str(x) for x in range(len(block_dims))]
    
    # 3. Construct Sparse Matrices
    # We need to build a single sparse matrix 'A_mat' of shape (m, total_dim).
    # And a sparse vector 'C_vec' of shape (total_dim,).
    
    # Arrays for triplet construction
    A_rows = []
    A_cols = []
    A_vals = []
    
    C_rows = [] # flattened indices
    C_cols = [] # all zeros (it's a vector)
    C_vals = []
    
    # Fi[0] is Objective (C)
    F0 = Fi[0]
    for i in range(len(F0.value)):
        blk = F0.block_number[i] - 1
        r = F0.row[i] - 1
        c = F0.column[i] - 1
        val = F0.value[i]
        
        dim = abs(block_dims[blk])
        flat_idx = block_offsets[blk] + r * dim + c
        
        C_rows.append(flat_idx)
        C_vals.append(val)
        
    # Fi[1..m] are Constraints (A)
    for k in range(m):
        Fk = Fi[k+1]
        for i in range(len(Fk.value)):
            blk = Fk.block_number[i] - 1
            r = Fk.row[i] - 1
            c = Fk.column[i] - 1
            val = Fk.value[i]
            
            dim = abs(block_dims[blk])
            flat_idx = block_offsets[blk] + r * dim + c
            
            A_rows.append(k)
            A_cols.append(flat_idx)
            A_vals.append(val)

    # 4. Create JAX Sparse Arrays
    # C vector
    if len(C_rows) > 0:
        C_indices = jnp.array(C_rows)[:, None] # (nnz, 1)
        C_values = jnp.array(C_vals)
        C_vec = sparse.BCOO((C_values, C_indices), shape=(total_dim,))
    else:
        C_vec = sparse.BCOO((jnp.empty(0), jnp.empty((0,1), dtype=int)), shape=(total_dim,))

    # A matrix
    if len(A_rows) > 0:
        A_indices = jnp.column_stack((jnp.array(A_rows), jnp.array(A_cols))) # (nnz, 2)
        A_values = jnp.array(A_vals)
        A_mat = sparse.BCOO((A_values, A_indices), shape=(m, total_dim))
    else:
        A_mat = sparse.BCOO((jnp.empty(0), jnp.empty((0,2), dtype=int)), shape=(m, total_dim))

    # --- Constraint Scaling (Ruiz Equilibration / Diagonal Preconditioning) ---
    # Compute the Frobenius norm of each constraint matrix (row of A_mat)
    # Since each row represents a symmetric matrix mapped to a vector, the Frobenius norm
    # of the matrix is simply the L2 norm of the vector.
    if len(A_rows) > 0:
        # Compute row norms using the sparse representation
        row_sq_norms = jax.ops.segment_sum(A_values**2, jnp.array(A_rows), num_segments=m)
        row_norms = jnp.sqrt(row_sq_norms)
        
        # Prevent division by zero
        row_norms = jnp.where(row_norms < 1e-15, 1.0, row_norms)
        
        # Scale factor for each row is 1 / norm
        scale_factors = 1.0 / row_norms
        
        # Scale A_values
        # A_rows maps each non-zero element to its row index, so we can use it to index scale_factors
        scaled_A_values = A_values * scale_factors[jnp.array(A_rows)]
        
        # Scale b
        b = b * scale_factors
        
        # Re-build scaled A_mat
        A_mat = sparse.BCOO((scaled_A_values, A_indices), shape=(m, total_dim))

    # Create dummy X to get unravel function
    dummy_X = {name: np.zeros(shape) for name, shape in zip(block_names, shapes)}
    _, unravel_fn = ravel_dict_ordered(dummy_X, block_names)

    def flatten_X(X, unravel=False):
        # Flatten Pytree X using ravel_dict_ordered
        flat_X, unravel = ravel_dict_ordered(X, block_names)
        return flat_X, unravel

    def _AX_apply(A, X):
        x_flat, _ = flatten_X(X)
        return {'y': A @ x_flat}
    AX_fn = jax.tree_util.Partial(_AX_apply, A_mat)
    # def AX_fn(X):
    #     x_flat, _ = flatten_X(X)
    #     return {'y': A_mat @ x_flat}
    def _ATy_apply_implicit(A, y):
        flat_x = y['y'] @ A
        return unravel_fn(flat_x)
    ATy_fn = jax.tree_util.Partial(_ATy_apply_implicit, A_mat)

    def _CX_apply(C, X):
        x_flat, _ = flatten_X(X)
        indices = C.indices.flatten()
        vals = C.data
        return jnp.dot(vals, x_flat[indices])
    CX_fn = jax.tree_util.Partial(_CX_apply, C_vec)

    if return_matrices:
        return AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X, A_mat, C_vec
    return AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X


def generate_problem_from_data_right_left_conditioned(b_list, Fi, block_dims, block_names=None, return_matrices=False):
    """
    Generates generic AX/CX operators for ADMM from parsed SDPA data.

    Args:
        b_list (list): Constraint vector elements.
        Fi (list): List of F blocks (F0 is objective, F1...Fm are constraints).
        block_dims (list): List of block dimensions.
        block_names (list): list of names of blocks in sequential order (for unraveling pytree)

    Returns:
        AX_fn (callable): Function mapping X (list of blocks) -> b (vector).
        CX_fn (callable): Function mapping X (list of blocks) -> scalar.
        b (jnp.array): constraint vector.
        shapes (list): List of shapes for the blocks of X.
    """
    b = jnp.array(b_list)
    m = len(b)
    
    # 2. Block Indexing & Offsets
    # We flatten X = [Block1, Block2, ...] into one long vector.
    
    block_sizes = [abs(d)**2 for d in block_dims]
    block_offsets = [0]
    for size in block_sizes:
        block_offsets.append(block_offsets[-1] + size)
    
    total_dim = block_offsets[-1]
    
    # Shapes for reconstruction/initializing X
    shapes = [(abs(d), abs(d)) for d in block_dims]

    if block_names is None:
        block_names = [str(x) for x in range(len(block_dims))]
    
    # 3. Construct Sparse Matrices
    # We need to build a single sparse matrix 'A_mat' of shape (m, total_dim).
    # And a sparse vector 'C_vec' of shape (total_dim,).
    
    # Arrays for triplet construction
    A_rows = []
    A_cols = []
    A_vals = []
    
    C_rows = [] # flattened indices
    C_cols = [] # all zeros (it's a vector)
    C_vals = []
    
    # Fi[0] is Objective (C)
    F0 = Fi[0]
    for i in range(len(F0.value)):
        blk = F0.block_number[i] - 1
        r = F0.row[i] - 1
        c = F0.column[i] - 1
        val = F0.value[i]
        
        dim = abs(block_dims[blk])
        flat_idx = block_offsets[blk] + r * dim + c
        
        C_rows.append(flat_idx)
        C_vals.append(val)
        
    # Fi[1..m] are Constraints (A)
    for k in range(m):
        Fk = Fi[k+1]
        for i in range(len(Fk.value)):
            blk = Fk.block_number[i] - 1
            r = Fk.row[i] - 1
            c = Fk.column[i] - 1
            val = Fk.value[i]
            
            dim = abs(block_dims[blk])
            flat_idx = block_offsets[blk] + r * dim + c
            
            A_rows.append(k)
            A_cols.append(flat_idx)
            A_vals.append(val)

    # 4. Create JAX Sparse Arrays
    # C vector
    if len(C_rows) > 0:
        C_indices = jnp.array(C_rows)[:, None] # (nnz, 1)
        C_values = jnp.array(C_vals)
        original_C_vec = sparse.BCOO((C_values, C_indices), shape=(total_dim,))
    else:
        original_C_vec = sparse.BCOO((jnp.empty(0), jnp.empty((0,1), dtype=int)), shape=(total_dim,))

    # A matrix
    if len(A_rows) > 0:
        A_indices = jnp.column_stack((jnp.array(A_rows), jnp.array(A_cols))) # (nnz, 2)
        A_values = jnp.array(A_vals)
        A_mat = sparse.BCOO((A_values, A_indices), shape=(m, total_dim))
    else:
        A_mat = sparse.BCOO((jnp.empty(0), jnp.empty((0,2), dtype=int)), shape=(m, total_dim))

    # --- Constraint Scaling (Ruiz Equilibration / Diagonal Preconditioning) ---
    # Compute the Frobenius norm of each constraint matrix (row of A_mat)
    # Since each row represents a symmetric matrix mapped to a vector, the Frobenius norm
    # of the matrix is simply the L2 norm of the vector.
    if len(A_rows) > 0:
        # Compute row norms using the sparse representation
        row_sq_norms = jax.ops.segment_sum(A_values**2, jnp.array(A_rows), num_segments=m)
        row_norms = jnp.sqrt(row_sq_norms)
        
        # Prevent division by zero
        row_norms = jnp.where(row_norms < 1e-15, 1.0, row_norms)
        
        # Scale factor for each row is 1 / norm
        scale_factors = 1.0 / row_norms
        
        # Scale A_values
        # A_rows maps each non-zero element to its row index, so we can use it to index scale_factors
        scaled_A_values = A_values * scale_factors[jnp.array(A_rows)]
        
        # Scale b
        b = b * scale_factors

    # =====================================================================
    # NEW: Block-wise Column Scaling (Right Preconditioning)
    # =====================================================================
    
    # 1. Map every flat column index to its corresponding Block ID
    col_to_block = jnp.zeros(total_dim, dtype=int)
    for k, size in enumerate(block_sizes):
        start = block_offsets[k]
        end = block_offsets[k+1]
        col_to_block = col_to_block.at[start:end].set(k)
        
    # Get the block ID for every non-zero element in A
    A_block_indices = col_to_block[jnp.array(A_cols)]
    
    # 2. Compute the average column norm for each block
    # Sum of squares of all elements belonging to block k
    block_sq_norms = jax.ops.segment_sum(scaled_A_values**2, A_block_indices, num_segments=len(block_dims))
    
    # Divide by the number of columns in the block to get the average col norm
    block_avg_col_norms = jnp.sqrt(block_sq_norms / jnp.array(block_sizes))
    
    # 3. Create the block scaling factors
    block_scale_factors = 1.0 / jnp.where(block_avg_col_norms < 1e-15, 1.0, block_avg_col_norms)
    
    # 4. Scale the A matrix values
    scaled_A_values = scaled_A_values * block_scale_factors[A_block_indices]
    
    # 5. Build the final, perfectly equilibrated A matrix
    A_mat = sparse.BCOO((scaled_A_values, A_indices), shape=(m, total_dim)) 


    # =====================================================================
    # CRITICAL: We must also scale the Objective Function (C)
    # Because we scaled the variables X -> X * block_scale_factors
    # =====================================================================
    if len(C_rows) > 0:
        C_block_indices = col_to_block[C_indices.flatten()]
        scaled_C_values = C_values * block_scale_factors[C_block_indices]
        C_vec = sparse.BCOO((scaled_C_values, C_indices), shape=(total_dim,))
    # =====================================================================

    # Create dummy X to get unravel function
    dummy_X = {name: np.zeros(shape) for name, shape in zip(block_names, shapes)}
    _, unravel_fn = ravel_dict_ordered(dummy_X, block_names)

    def flatten_X(X, unravel=False):
        # Flatten Pytree X using ravel_dict_ordered
        flat_X, unravel = ravel_dict_ordered(X, block_names)
        return flat_X, unravel

    def _AX_apply(A, X):
        x_flat, _ = flatten_X(X)
        return {'y': A @ x_flat}
    AX_fn = jax.tree_util.Partial(_AX_apply, A_mat)
    # def AX_fn(X):
    #     x_flat, _ = flatten_X(X)
    #     return {'y': A_mat @ x_flat}
    def _ATy_apply_implicit(A, y):
        flat_x = y['y'] @ A
        return unravel_fn(flat_x)
    ATy_fn = jax.tree_util.Partial(_ATy_apply_implicit, A_mat)

    def _CX_apply(C, X):
        x_flat, _ = flatten_X(X)
        indices = C.indices.flatten()
        vals = C.data
        return jnp.dot(vals, x_flat[indices])
    # CX_fn = jax.tree_util.Partial(_CX_apply, C_vec)

    # The solver uses the SCALED C
    CX_fn= jax.tree_util.Partial(_CX_apply, C_vec)
    
    # We return the TRUE C for physical evaluation at the end
    CX_fn_true = jax.tree_util.Partial(_CX_apply, original_C_vec)

    unscale_fn = lambda x, y: unscale_UV(x, y, block_scale_factors, block_names)

    if return_matrices:
        return AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X, A_mat, C_vec, block_scale_factors, unscale_fn, CX_fn_true
    return AX_fn, ATy_fn, CX_fn, b, shapes, block_names, flatten_X, block_scale_factors, unscale_fn, CX_fn_true

def unscale_UV(U_sol, V_sol, block_scale_factors, block_names):
    # 2. Un-scale U and V to get the physical SOS factors
    # block_names tells us the order. We just zip them with the scale factors.
    U_physical = {}
    V_physical = {}
    
    for block_name, scale_factor in zip(block_names, block_scale_factors):
        # We multiplied A by scale_factor, which means G_solver = G_true / scale_factor
        # So G_true = G_solver * scale_factor
        # Therefore, U_true = U_solver * sqrt(scale_factor)
        
        correction = jnp.sqrt(scale_factor)
        
        U_physical[block_name] = U_sol[block_name] * correction
        V_physical[block_name] = V_sol[block_name] * correction
    return U_physical, V_physical
