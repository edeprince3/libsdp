"""Test the sparse AX/CX operators and block layout used as ADMM inputs.

These tests exercise ``AX_CX_generator_from_sparse`` against Python BPSDP
and C++ solver references, including a PySCF-generated v2RDM problem.
They do not call ``run_admm`` or test ADMM iterations or convergence.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


import sys
import os
import numpy as np

import pytest
# import importlib.util

import libsdp
from libsdp.sdpa_file_io import read_sdpa_problem, write_sdpa_problem, clean_sdpa_problem
from libsdp.bpsdp import evaluate_Au
from libsdp.AX_CX_generator_from_sparse import generate_problem_from_sdpa, generate_problem_from_data
from libsdp.sdp_helper import sdp_solver
from libsdp import sdp_options
from libsdp.jax_utilities import ravel_dict_ordered


def _constraint_row_norms(Fi):
    """Norms of the unscaled reference constraints (F0 is the objective)."""
    norms = np.array([np.linalg.norm(F.value) for F in Fi[1:]])
    return np.where(norms < 1e-15, 1.0, norms)


def test_vectorization_indexing():
    """
    Verifies that the JAX flattening logic matches the reference manual indexing
    used in libsdp.bpsdp.solve.
    """
    block_dims = [2, 3]
    offsets = [0, 4]
    
    test_cases = [
        (1, 1, 1, 0),
        (1, 2, 1, 2),
        (2, 1, 1, 4),
        (2, 3, 2, 11)
    ]
    
    print("\n[Test] Checking Vectorization Indexing...")
    for b, r, c_idx, expected in test_cases:
        blk_0 = b - 1
        r_0 = r - 1
        c_0 = c_idx - 1
        
        dim = block_dims[blk_0]
        calculated = offsets[blk_0] + r_0 * dim + c_0
        assert calculated == expected, f"Calculated {calculated} != Expected {expected}"

def test_ax_vs_bpsdp_reference():
    """
    Compares normalized JAX AX and b with the unscaled BPSDP reference.
    """
    libsdp_examples_path = os.path.join(os.path.dirname(libsdp.__path__[0]), 'examples')
    candidate_paths = [
        os.path.join(libsdp_examples_path, "python_interface/truss1.dat-s"),
    ]
    
    filename = None
    for p in candidate_paths:
        if os.path.exists(p):
            filename = p
            break
            
    if not filename:
        print(f"Skipping truss1 test: truss1.dat-s not found in {candidate_paths}")
        return

    print(f"\n[Test] Verifying AX against bpsdp.evaluate_Au using {filename}...")

    # 1. Generate JAX operators
    AX_fn, ATy_fn, CX_fn, b, shapes, pytree_ordered_node_names, _ = generate_problem_from_sdpa(filename)
    
    # 2. Setup Reference Data
    b_ref, Fi, block_dim = read_sdpa_problem(filename)
    row_norms = _constraint_row_norms(Fi)
    np.testing.assert_allclose(b, np.asarray(b_ref) / row_norms, atol=1e-8)
    
    # Fi contains [Objective, Constraint1, Constraint2...]
    # We need to manually populate the .id fields for bpsdp.evaluate_Au
    primal_block_dim = block_dim 
    
    def get_block_offset(blk_idx):
        off = 0
        for k in range(blk_idx):
            off += int(abs(primal_block_dim[k])**2)
        return off

    Ai_constraints = []
    for i in range(1, len(Fi)):
        F_orig = Fi[i]
        F_orig.id = [] 
        
        for j in range(len(F_orig.block_number)):
            my_block = F_orig.block_number[j] - 1
            my_row = F_orig.row[j] - 1
            my_column = F_orig.column[j] - 1
            
            off = get_block_offset(my_block)
            dim = abs(primal_block_dim[my_block])
            
            idx = off + my_row * dim + my_column
            F_orig.id.append(idx)
            
        Ai_constraints.append(F_orig)
    
    # 3. Create Random X
    np.random.seed(42)
    X_list_np = []
    X_list_jax = []
    X_pytree = {} 
    for idx, s in enumerate(shapes):
        mat = np.random.randn(s[0], s[1])
        mat = 0.5 * (mat + mat.T)
        X_list_np.append(mat)
        X_list_jax.append(jnp.array(mat))
        X_pytree[str(idx)] = X_list_jax[-1]
        
    x_flat_np = np.concatenate([m.reshape(-1) for m in X_list_np])
    
    # 4. Evaluate Reference
    ax_ref = evaluate_Au(Ai_constraints, x_flat_np) / row_norms
    
    # 5. Evaluate JAX
    # ax_jax = AX_fn(X_list_jax)['y']
    ax_pytree = AX_fn(X_pytree)['y']

    # 6. Compare
    print("Reference (First 5):", ax_ref[:5])
    # print("JAX       (First 5):", ax_jax[:5])
    print("PYTREE    (First 5):", ax_pytree[:5])
   
    #  error = np.linalg.norm(ax_ref - ax_jax)
    error_pytree = np.linalg.norm(ax_ref - ax_pytree)
    # print(f"L2 Difference: {error:.2e}")
    print(f"L2 Difference - pytree: {error_pytree:.2e}")
    
    assert np.allclose(ax_ref, ax_pytree, atol=1e-8), "JAX AX does not match bpsdp.evaluate_Au"
    print("SUCCESS: JAX AX matches bpsdp.evaluate_Au.")

def test_ax_cx_against_cpp_solver():
    """
    Compares normalized AX/b and unscaled CX against the C++ reference.
    """
    libsdp_path = libsdp.__path__[0]
    libsdp_examples_path = os.path.join(os.path.dirname(libsdp_path), 'examples')
    print(f"libsdp examples path: {libsdp_examples_path}")
    candidate_paths = [
        os.path.join(libsdp_examples_path, "python_interface/truss1.dat-s"),
    ]
    
    filename = None
    for p in candidate_paths:
        if os.path.exists(p):
            filename = p
            break
            
    if not filename:
        print(f"Skipping cpp solver test: truss1.dat-s not found in {candidate_paths}")
        return

    print(f"\n[Test] Verifying AX/CX against C++ sdp_solver using {filename}...")

    # 1. Generate JAX operators
    AX_fn, ATy_fn, CX_fn, b_vec, shapes, pytree_ordered_node_names, _ = generate_problem_from_sdpa(filename)
    
    # 2. Setup C++ Solver
    b_ref, A_ref, block_dim_ref = read_sdpa_problem(filename)
    row_norms = _constraint_row_norms(A_ref)
    np.testing.assert_allclose(b_vec, np.asarray(b_ref) / row_norms, atol=1e-8)
    options = sdp_options()
    # Minimal options to avoid output/overhead if possible, though constructor doesn't take much
    solver = sdp_solver(options, A_ref, block_dim_ref)
    
    # 3. Create Random X
    np.random.seed(123)
    X_list_np = []
    X_list_jax = []
    X_pytree = {} 
    pytree_ordered_node_names = []
    for idx, s in enumerate(shapes):
        mat = np.random.randn(s[0], s[1])
        mat = 0.5 * (mat + mat.T) # Make symmetric
        X_list_np.append(mat)
        X_list_jax.append(jnp.array(mat))
        X_pytree[str(idx)] = X_list_jax[-1]
        pytree_ordered_node_names.append(str(idx))

    # Flatten X for C++ solver (assumes standard layout)
    x_flat_np = np.concatenate([m.reshape(-1) for m in X_list_np])

    # 4. Compare AX
    # C++: sdp.get_Au(x)
    ax_cpp = np.array(solver.get_Au(x_flat_np)) / row_norms
    # JAX: AX_fn(X)
    # ax_jax = AX_fn(X_list_jax)['y']
    # exit()
    ax_pytree = AX_fn(X_pytree)['y']
    
    print("AX C++    (First 5):", ax_cpp[:5])
    # print("AX JAX    (First 5):", ax_jax[:5])
    print("AX pytree (First 5):", ax_pytree[:5])

    # ax_error = np.linalg.norm(ax_cpp - ax_jax)
    # print(f"AX L2 Difference: {ax_error:.2e}")
    # assert np.allclose(ax_cpp, ax_jax, atol=1e-8), "JAX AX does not match sdp_solver.get_Au"
    ax_error = np.linalg.norm(ax_cpp - ax_pytree)
    print(f"AX-pytree L2 Difference: {ax_error:.2e}")
    assert np.allclose(ax_cpp, ax_pytree, atol=1e-8), "JAX AX does not match sdp_solver.get_Au"
    
    # 5. Compare CX
    # C++: sdp.get_c() returns flats vector C. Objective is typically dot(C, X).
    c_vec_cpp = np.array(solver.get_c())
    cx_cpp = np.dot(c_vec_cpp, x_flat_np)
    
    # JAX: CX_fn(X)
    # cx_jax = CX_fn(X_list_jax)
    cx_pytree = CX_fn(X_pytree)
    
    print(f"CX C++:    {cx_cpp:.12f}")
    # print(f"CX JAX:    {cx_jax:.12f}")
    print(f"CX pytree: {cx_pytree:.12f}")
    
    cx_error = abs(cx_cpp - cx_pytree)
    print(f"CX Difference: {cx_error:.2e}")
    assert np.isclose(cx_cpp, cx_pytree, atol=1e-8), "JAX CX does not match sdp_solver.get_c dot X"
    
    print("SUCCESS: JAX AX/CX matches C++ sdp_solver.")

    print("SUCCESS: JAX AX/CX matches C++ sdp_solver.")

def test_pyscf_v2rdm_ax_cx():
    """
    Generates an SDP using PySCF + v2rdm_sdp, writes to file, 
    and verifies AX/CX against libsdp on the generated problem.
    """
    pyscf = pytest.importorskip("pyscf", reason="PySCF is optional and required for this integration test")
    print("\n[Test] Generating SDP from PySCF (H6 chain)...")
    
    # 1. Setup paths to import v2rdm_sdp
    libsdp_path = libsdp.__path__[0]
    # Assuming layout: libsdp/libsdp/test_admm_operators.py
    # examples/pyscf_interface is at ../../examples/pyscf_interface relative to this file?
    # Actually, we rely on finding it relative to libsdp package location or cwd.
    # The user said "examples/pyscf_interface/pyscf_v2rdm.py" exists.
    # Let's verify location relative to libsdp path.
    # If libsdp is installed in site-packages, examples might not be there.
    # But usually for dev installs they are.
    # Let's try to find it in the dev workspace path we inferred earlier.
    
    # We used os.path.dirname(libsdp_path) + '/examples' before.
    libsdp_examples_path = os.path.join(os.path.dirname(libsdp_path), 'examples')
    pyscf_interface_path = os.path.join(libsdp_examples_path, 'pyscf_interface')
    
    if not os.path.exists(pyscf_interface_path):
        print(f"Skipping PySCF test: {pyscf_interface_path} does not exist.")
        return
        
    sys.path.append(pyscf_interface_path)
    
    try:
        from v2rdm_sdp import v2rdm_sdp
    except ImportError:
        print("Skipping PySCF test: Could not import v2rdm_sdp from examples.")
        sys.path.pop()
        return

    # 2. Build Molecule & Integrals (Copied from pyscf_v2rdm.py)
    mol = pyscf.M(
        atom=[
            ['H', (0, 0, 0)],
            ['H', (0, 0, 1)],
            # ['H', (0, 0, 2)],
            # ['H', (0, 0, 3)],
            # ['H', (0, 0, 4)],
            # ['H', (0, 0, 5)],
        ],
        basis='sto-3g',
        spin=0,
        charge=0,
        symmetry=False,
        verbose=0
    )

    mf = mol.RHF().run()
    C = mf.mo_coeff
    tei = mol.intor('int2e')
    tei = np.einsum('uj,vi,wl,xk,uvwx', C, C, C, C, tei, optimize=True)
    
    kinetic = mol.intor('int1e_kin')
    potential = mol.intor('int1e_nuc')
    oei = kinetic + potential
    oei = np.einsum('uj,vi,uv', C, C, oei)

    nmo = int(mf.mo_coeff.shape[1])
    occ = mf.mo_occ
    nele = int(sum(occ))
    nalpha = nele // 2
    nbeta = nalpha     
    my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2=True, g2=True, constrain_spin=False)
    
    # 3. Clean and Write SDP
    b, F = clean_sdpa_problem(my_sdp.b, my_sdp.F)
    dimensions = my_sdp.dimensions
    options = sdp_options()
    options.sdp_algorithm             = "bpsdp"
    options.guess_type                = "random"
    options.sdp_error_convergence     = 5e-5
    options.sdp_objective_convergence = 5e-5
    options.cg_convergence            = 1e-8
    options.dynamic_cg_convergence    = False
    # solve sdp
    maxiter = 100000
    solver = sdp_solver(options, F, dimensions)
    x = jnp.array(solver.solve(b, maxiter))
    x_blocks = my_sdp.get_rdm_blocks(x)
    x_blocks_jax = [jnp.array(xx) for xx in x_blocks]
    x_blocks_pytree = dict((my_sdp.blocks[xx], x_blocks_jax[xx]) for xx in range(len(x_blocks)))

    flat_X, unravel_func = ravel_dict_ordered(x_blocks_pytree, my_sdp.blocks)
    assert jnp.allclose(flat_X, x)
    test_x_pytree = unravel_func(flat_X)
    for key in my_sdp.blocks:
        assert jnp.allclose(test_x_pytree[key], x_blocks_pytree[key])
    
    test_filename = 'temp_h6_chain.dat-s'
    print(f"Writing temporary SDP file to {test_filename}...")
    write_sdpa_problem(test_filename, b, F, dimensions)
    
    # 4. Generate JAX operators
    AX_fn, ATy_fn, CX_fn, b_jax, shapes, pytree_ordered_node_names, _ = generate_problem_from_data(b, F, dimensions, my_sdp.blocks)
    row_norms = _constraint_row_norms(F)
    np.testing.assert_allclose(b_jax, np.asarray(b) / row_norms, atol=1e-8)
   
    # 6. Compare on Random Vector
    print("[Test] Verifying Operators on Random Vector...")
    np.random.seed(42)
    X_list_np = []
    X_list_jax = []
    X_pytree = {}
    for idx, s in enumerate(shapes):
        mat = np.random.randn(s[0], s[1])
        mat = 0.5 * (mat + mat.T)
        X_list_np.append(mat)
        X_list_jax.append(jnp.array(mat))
        X_pytree[my_sdp.blocks[idx]] = X_list_jax[-1]

    x_flat_np = np.concatenate([m.reshape(-1) for m in X_list_np])
    
    # AX
    ax_cpp = np.array(solver.get_Au(x_flat_np)) / row_norms
    ax_pytree = AX_fn(X_pytree)['y']
   
    ax_diff = np.linalg.norm(ax_cpp - ax_pytree)
    print(f"AX L2 Diff: {ax_diff:.2e}")
    assert ax_diff < 1e-8, "AX mismatch on random vector in PySCF test"
    ax_cpp = np.array(solver.get_Au(x)) / row_norms
    ax_pytree = AX_fn(x_blocks_pytree)['y']
    ax_diff = np.linalg.norm(ax_cpp - ax_pytree)
    print(f"AX L2 Diff: {ax_diff:.2e}")
    
    # CX
    c_vec_cpp = np.array(solver.get_c())
    # cx_cpp = np.dot(c_vec_cpp, x_flat_np)
    cx_cpp = np.dot(c_vec_cpp, x)
    # cx_jax = CX_fn(X_list_jax)
    cx_jax = CX_fn(x_blocks_pytree)
    print(cx_jax)
    
    cx_diff = abs(cx_cpp - cx_jax)
    print(f"CX Diff:    {cx_diff:.2e}")
    print(f"{cx_jax=}")
    
    # Cleanup
    if os.path.exists(test_filename):
        os.remove(test_filename)
        
    sys.path.pop() # Restore sys.path
    
    assert ax_diff < 1e-8, "AX mismatch in PySCF test"
    assert cx_diff < 1e-8, "CX mismatch in PySCF test"
    print("SUCCESS: PySCF v2RDM AX/CX Verified.")

    X_list_jax = x_blocks_jax
    x_flat_np = np.concatenate([m.reshape(-1) for m in x_blocks])
 
    # AX
    ax_cpp = np.array(solver.get_Au(x_flat_np)) / row_norms
    ax_jax = AX_fn(x_blocks_pytree)['y']
    
    ax_diff = np.linalg.norm(ax_cpp - ax_jax)
    print(f"AX L2 Diff: {ax_diff:.2e}")
    
    # CX
    c_vec_cpp = np.array(solver.get_c())
    cx_cpp = np.dot(c_vec_cpp, x_flat_np)
    cx_jax = CX_fn(x_blocks_pytree)
    
    cx_diff = abs(cx_cpp - cx_jax)
    print(f"CX Diff:    {cx_diff:.2e}")
    
    # Cleanup
    if os.path.exists(test_filename):
        os.remove(test_filename)
        
    sys.path.pop() # Restore sys.path
    
    assert ax_diff < 1e-8, "AX mismatch in PySCF test"
    assert cx_diff < 1e-8, "CX mismatch in PySCF test"
    print("SUCCESS: PySCF v2RDM AX/CX Verified.")


def test_generate_from_data_direct():
    """
    Verifies that generate_problem_from_data works correctly with manually passed data.
    """
    print("\n[Test] Verifying generate_problem_from_data with manual data...")
    
    # 1. Setup Manual Data
    # Simple problem: min Tr(C X) s.t. Tr(A1 X) = b1
    # Block dims: [2] (one 2x2 block)
    # C = [[1, 0], [0, 1]] (Identity)
    # A1 = [[2, 0], [0, 2]]
    # b1 = 4
    # X = [[1, 0], [0, 1]] -> Tr(CX) = 2, Tr(A1X) = 4. Satisfied.
    
    b_list = [4.0]
    block_dims = [2]
    
    # Mocking F (SDPA format uses 1-based indexing)
    # F0 (Objective C)
    class MockF:
        def __init__(self, block_number, row, column, value):
            self.block_number = block_number
            self.row = row
            self.column = column
            self.value = value

    # C = I. Diagonals (1,1) and (2,2) for block 1.
    F0 = MockF(
        block_number=[1, 1],
        row=[1, 2],
        column=[1, 2],
        value=[1.0, 1.0]
    )
    
    # A1 (Constraint 1)
    F1 = MockF(
        block_number=[1, 1],
        row=[1, 2],
        column=[1, 2],
        value=[2.0, 2.0]
    )
    
    Fi = [F0, F1]
    
    # 2. Generate Operators
    AX_fn, ATy_fn, CX_fn, b, shapes, pytree_ordered_node_names, _ = generate_problem_from_data(b_list, Fi, block_dims)
    
    # 3. Test on X = I
    X_list = {'0': jnp.eye(2)}
    
    # ||A1||_F = sqrt(8), so both AX(I) and b are 4 / sqrt(8) = sqrt(2).
    expected_ax = jnp.array([np.sqrt(2.0)])
    ax_val = AX_fn(X_list)['y']
    print(f"AX([I]): {ax_val}")
    np.testing.assert_allclose(ax_val, expected_ax, atol=1e-8)
    np.testing.assert_allclose(b, expected_ax, atol=1e-8)
    
    # CX should be 2.0
    cx_val = CX_fn(X_list)
    print(f"CX([I]): {cx_val}")
    assert jnp.allclose(cx_val, 2.0), f"CX failed: {cx_val} != 2.0"
    
    print("SUCCESS: generate_problem_from_data verified with manual data.")

if __name__ == "__main__":
    test_vectorization_indexing()
    test_ax_vs_bpsdp_reference()
    test_ax_cx_against_cpp_solver()
    test_generate_from_data_direct()
    test_pyscf_v2rdm_ax_cx()

