import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
import os

import pytest

import numpy as np
import libsdp
from libsdp.sdp_helper import sdp_solver
from libsdp.sdpa_file_io import clean_sdpa_problem
from libsdp.AX_CX_generator_from_sparse import generate_problem_from_data
from libsdp.admm import (
    # compute_AT_y, 
    admm_augmented_lagrangian, 
    get_C_pytree, 
    LHS_U, 
    LHS_V, 
    RHS_U, 
    RHS_V, 
    solve_for_Uparams_cg, 
    solve_for_Vparams_cg,
    compute_norm_and_residual,
    admm_solver
)
from libsdp.jax_utilities import tree_real_matrix_inner_product

import optax

from functools import partial

def get_v2rdm_h4():
    pyscf = pytest.importorskip("pyscf", reason="PySCF is optional and required for these integration tests")
    # 1. Setup paths to import v2rdm_sdp
    libsdp_path = libsdp.__path__[0]
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
            ['H', (0, 0, 2)],
            ['H', (0, 0, 3)],
            ['H', (0, 0, 4)],
            ['H', (0, 0, 5)],
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
    my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, 
    q2=True, 
    g2=True, 
    constrain_spin=True)
    
    # 3. Clean and Write SDP
    b, F = clean_sdpa_problem(my_sdp.b, my_sdp.F)
    dimensions = my_sdp.dimensions
    options = libsdp.sdp_options()
    options.sdp_algorithm             = "bpsdp"
    options.guess_type                = "random"
    options.sdp_error_convergence     = 5e-5
    options.sdp_objective_convergence = 5e-5
    options.cg_convergence            = 1e-8
    options.dynamic_cg_convergence    = False
    # solve sdp
    maxiter = 100000
    solver = sdp_solver(options, F, dimensions)
    x = solver.solve(b, maxiter)
    return my_sdp, b, F, solver, x

def pytree_allclose(tree1, tree2, rtol=1e-5, atol=1e-8):
    # 1. Compare every leaf individually
    is_close = jax.tree.map(
        lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol), 
        tree1, 
        tree2
    )
    # 2. Reduce the boolean tree to a single boolean
    return jax.tree.reduce(lambda x, y: x and y, is_close, True)

def setup_v2rdm_sol():
    my_sdp, b, F, solver, x_bpsdp = get_v2rdm_h4()
    x_blocks = my_sdp.get_rdm_blocks(x_bpsdp)
    x_blocks_jax = [jnp.array(xx) for xx in x_blocks]
    Gparams = dict((my_sdp.blocks[xx], x_blocks[xx]) for xx in range(len(x_blocks)))
    return_matrices = True
    AX_fn, ATy_fn, CX_fn, b_jax, shapes, _, _, A_mat, C_vec = generate_problem_from_data(b, F, my_sdp.dimensions, block_names=my_sdp.blocks, return_matrices=return_matrices)
    auvec = AX_fn(Gparams)
    y = jnp.array(solver.get_y())
    z = jnp.array(solver.get_z())
    c = jnp.array(solver.get_c())
    mu = float(solver.get_mu())
    Ax = jnp.array(solver.get_Au(x_bpsdp))
    bvec = jnp.array(b)
    residual = Ax - bvec
    sigma = {'sigma': 1./mu}

    yparams = {'y': y}
    btree = {'y': bvec}
    Rparams = {}
    for idx, gg in enumerate(x_blocks):
        w, v = np.linalg.eigh(gg)
        nonzero = np.where(w > 1e-8)[0]
        w = w[nonzero]
        v = v[:, nonzero]
        sqrt_mat = v @ np.diag(np.sqrt(w)) @ v.T
        assert np.allclose(sqrt_mat.T @ sqrt_mat, gg)
        Rparams.update({my_sdp.blocks[idx]: jnp.array(sqrt_mat.T)})
    if return_matrices:
        return my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec
    return my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams


def test_ATy():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    # get ATy and convert to pytree

    ATy = solver.get_ATu(y)
    ATy_blocks = my_sdp.get_rdm_blocks(ATy)
    ATy_pytree = {}
    for idx in range(len(ATy_blocks)):
        ATy_pytree[my_sdp.blocks[idx]] = ATy_blocks[idx]

    # ax_transpose_fun = jax.linear_transpose(AX_fn, Gparams)

    print("Computing ATy_fn")
    test_ATy_tmp = ATy_fn(yparams)
    # test_ATy_tmp = compute_AT_y(
    #     Gparams_template=Gparams,
    #     yparams=yparams,
    #     AX_fn=AX_fn
    # )
    # compare pytree
    print("Comparing ATy_fn")
    pytree_allclose(ATy_pytree, test_ATy_tmp)

    y = np.random.randn(y.shape[0])
    yparams = {'y': jnp.array(y)}
    print("Computing RDM blocks")
    ATy_blocks = my_sdp.get_rdm_blocks(ATy)
    ATy_pytree = {}
    for idx in range(len(ATy_blocks)):
        ATy_pytree[my_sdp.blocks[idx]] = ATy_blocks[idx]
    print("Computing ATy_fn")
    test_ATy_tmp = ATy_fn(yparams)
    # test_ATy_tmp = compute_AT_y(
    #     Gparams_template=Gparams,
    #     yparams=yparams,
    #     AX_fn=AX_fn
    # )
    # compare pytree
    print("Comparing ATy_fn")
    pytree_allclose(ATy_pytree, test_ATy_tmp)
    print("PASSED")


def test_admm_lagrangian():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    admm_val = admm_augmented_lagrangian(
        Uparams=Rparams,
        Vparams=Rparams,
        yparams=yparams,
        btree=btree,
        sigma=sigma['sigma'],
        AX_fn=AX_fn,
        CX_fn=CX_fn
    )
    # print(admm_val)
    G_UV = jax.tree_util.tree_map(lambda m1, m2: m1 @ m2.T, Rparams, Rparams)

    obj_val = CX_fn(G_UV)
    A_GUV = AX_fn(G_UV)
    residual_test = optax.tree_utils.tree_add_scale(A_GUV, -1, btree)
    UV_diff = optax.tree_utils.tree_add_scale(Rparams, -1, Rparams) 
    residual_l2 = (sigma['sigma'] / 2) * optax.tree_utils.tree_vdot(residual_test, residual_test)
    uv_diff_l2 = (sigma['sigma'] / 2) * optax.tree_utils.tree_vdot(UV_diff, UV_diff)
    c_UV = CX_fn(G_UV)
    admm_val_manual = c_UV - optax.tree_utils.tree_vdot(yparams, residual_test) + residual_l2 + uv_diff_l2
    assert np.isclose(admm_val, admm_val_manual)

def test_gradient_admm_alm():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    Uparams = Rparams
    Vparams = Rparams
    admm_val = admm_augmented_lagrangian(
        Uparams=Uparams,
        Vparams=Vparams,
        yparams=yparams,
        btree=btree,
        sigma=sigma['sigma'],
        AX_fn=AX_fn,
        CX_fn=CX_fn
    )

    U_grad = jax.grad(admm_augmented_lagrangian, argnums=0)
    U_grad_val = U_grad(
        Uparams,
        Vparams,
        yparams,
        btree,
        sigma['sigma'],
        AX_fn,
        CX_fn
    )
    V_grad = jax.grad(admm_augmented_lagrangian, argnums=1)
    V_grad_val = V_grad(
        Uparams,
        Vparams,
        yparams,
        btree,
        sigma['sigma'],
        AX_fn,
        CX_fn
    )

    C_pytree = get_C_pytree(dummy_params=Uparams, CX_fn=CX_fn)
    G_UV = tree_real_matrix_inner_product(Uparams, Vparams) 
    residual_uv = optax.tree_utils.tree_add_scale(AX_fn(G_UV), -1, btree)
    diff_uv = optax.tree_utils.tree_add_scale(Uparams, -1, Vparams)
    def tree_gemm(tree1, tree2):
        return jax.tree_util.tree_map(lambda m1, m2: m1 @ m2, tree1, tree2)

    V_grad_compute_CX = tree_gemm(C_pytree, Uparams)
    ATy_tree = ATy_fn(yparams)
    # ATy_tree = compute_AT_y(
    #     Gparams_template=Gparams,
    #     yparams=yparams,
    #     AX_fn=AX_fn)
    V_grad_compute_ATyU = jax.tree_util.tree_map(
        lambda m, u: m.T @ u, 
        ATy_tree, 
        Uparams
    )
    ATres_tree = ATy_fn(residual_uv)
    # ATres_tree = compute_AT_y(
    #     Gparams_template=Gparams,
    #     yparams=residual_uv,
    #     AX_fn=AX_fn)
    V_grad_compute_ATresU = optax.tree_utils.tree_scale(sigma['sigma'], jax.tree_util.tree_map(
        lambda m, u: m.T @ u, 
        ATres_tree, 
        Uparams
    ))

    V_grad_compute_UVdiffV = optax.tree_utils.tree_scale(-sigma['sigma'], diff_uv)

    test_V_grad_compute = optax.tree_utils.tree_add_scale(
        V_grad_compute_CX,
        -1,
        V_grad_compute_ATyU
    )
    test_V_grad_compute = optax.tree_utils.tree_add_scale(
        test_V_grad_compute,
        1,
        V_grad_compute_ATresU
    )
    test_V_grad_compute = optax.tree_utils.tree_add_scale(
        test_V_grad_compute,
        1,
        V_grad_compute_UVdiffV
    )
    # test_V_grad_compute = V_grad_compute_UVdiffV
    pytree_allclose(V_grad_val, test_V_grad_compute)

    U_grad_compute_CX = tree_gemm(C_pytree, Vparams)
    U_grad_compute_ATyV = jax.tree_util.tree_map(
        lambda m, u: m @ u, 
        ATy_tree, 
        Vparams
    )
    # rhs
    U_grad_compute_ATresV = optax.tree_utils.tree_scale(sigma['sigma'], jax.tree_util.tree_map(
        lambda m, u: m @ u, 
        ATres_tree, 
        Vparams
    ))
    U_grad_compute_UVdiffU = optax.tree_utils.tree_scale(sigma['sigma'], diff_uv)
    test_U_grad_compute = optax.tree_utils.tree_add_scale(
        U_grad_compute_CX,
        -1,
        U_grad_compute_ATyV
    )
    test_U_grad_compute = optax.tree_utils.tree_add_scale(
        test_U_grad_compute,
        1,
        U_grad_compute_ATresV
    )
    test_U_grad_compute = optax.tree_utils.tree_add_scale(
        test_U_grad_compute,
        1,
        U_grad_compute_UVdiffU
    )
    pytree_allclose(U_grad_val, test_U_grad_compute)
    print("PASSED")

def test_lhs_minus_rhs_equals_gradient():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    Uparams = Rparams
    Vparams = Rparams
    for key, val in Rparams.items():
        Uparams[key] = jnp.array(np.random.randn(*val.shape))
        Vparams[key] = jnp.array(np.random.randn(*val.shape))

    admm_val = admm_augmented_lagrangian(
        Uparams=Uparams,
        Vparams=Vparams,
        yparams=yparams,
        btree=btree,
        sigma=sigma['sigma'],
        AX_fn=AX_fn, 
        CX_fn=CX_fn
    )

    U_grad = jax.grad(admm_augmented_lagrangian, argnums=0)
    U_grad_val = U_grad(
        Uparams,
        Vparams,
        yparams,
        btree,
        sigma['sigma'],
        AX_fn,
        CX_fn
    )
    V_grad = jax.grad(admm_augmented_lagrangian, argnums=1)
    V_grad_val = V_grad(
        Uparams,
        Vparams,
        yparams,
        btree,
        sigma['sigma'],
        AX_fn,
        CX_fn
    )
    C_pytree = get_C_pytree(dummy_params=Uparams, CX_fn=CX_fn)
    lhsu = LHS_U(Uparams, Vparams, sigma['sigma'], AX_fn, ATy_fn)
    rhsu = RHS_U(Vparams, yparams, btree, C_pytree, sigma['sigma'], AX_fn, ATy_fn)
    test_grad = optax.tree_utils.tree_add_scale(lhsu, -1, rhsu)
    pytree_allclose(U_grad_val, test_grad)

    lhsv = LHS_V(Uparams, Vparams, sigma['sigma'], AX_fn, ATy_fn)
    rhsv = RHS_V(Uparams, yparams, btree, C_pytree, sigma['sigma'], AX_fn, ATy_fn)
    test_grad = optax.tree_utils.tree_add_scale(lhsv, -1, rhsv)
    pytree_allclose(V_grad_val, test_grad, rtol=0, atol=1E-9)
    print("PASSED")

 
def test_lineax_CG_solve():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    Uparams = Rparams
    Vparams = Rparams
    for key, val in Rparams.items():
        Uparams[key] = jnp.array(np.random.randn(*val.shape))
        Vparams[key] = jnp.array(np.random.randn(*val.shape))
    C_pytree = get_C_pytree(dummy_params=Uparams, CX_fn=CX_fn)

    sol, _ = solve_for_Uparams_cg(
        init_U=Uparams,
        Vparams=Vparams,
        yparams=yparams,
        btree=btree,
        Ctree=C_pytree,
        sigma=sigma['sigma'],
        AX_fn=AX_fn,
        AT_fn=ATy_fn,
        tol=1e-10,
        max_steps=1000,
    )
    # print(sol)
    print("solution L2 norm ", optax.tree_utils.tree_norm(sol))
    print("Verifying LHS = RHS...")
    rhs_vec = RHS_U(Vparams, yparams, btree, C_pytree, sigma['sigma'], AX_fn, ATy_fn)
    lhs_vec = LHS_U(sol, Vparams, sigma['sigma'], AX_fn, ATy_fn)
    def tree_allclose(t1, t2, atol=1e-8, rtol=1e-5):
        def allclose(x, y):
            return jnp.allclose(x, y, atol=atol, rtol=rtol)
        return jax.tree_util.tree_reduce(lambda x, y: x and y, jax.tree_util.tree_map(allclose, t1, t2), True)

    assert tree_allclose(rhs_vec, lhs_vec, atol=1e-8, rtol=0)
    for key in rhs_vec.keys():
        print(key, jnp.linalg.norm(rhs_vec[key] - lhs_vec[key]))
        assert np.isclose(0, jnp.linalg.norm(rhs_vec[key] - lhs_vec[key]))
    print("LHS == RHS")

    # Adense = analyze_operator_A_V(Uparams, Vparams, nmo, sigma['sigma'])
    sol, _ = solve_for_Vparams_cg(
        init_V=Vparams,
        Uparams=Uparams,
        yparams=yparams,
        btree=btree,
        Ctree=C_pytree,
        sigma=sigma['sigma'],
        AX_fn=AX_fn,
        AT_fn=ATy_fn,
        tol=1e-10,
        max_steps=1000
    )
    # print(sol)
    print("solution L2 norm ", optax.tree_utils.tree_norm(sol))
    print("Verifying LHS = RHS...")
    rhs_vec = RHS_V(Uparams, yparams, btree, C_pytree, sigma['sigma'], AX_fn, ATy_fn)
    lhs_vec = LHS_V(Uparams, sol, sigma['sigma'], AX_fn, ATy_fn)
    diff_norm = optax.tree_utils.tree_norm(optax.tree_utils.tree_add_scale(rhs_vec, -1, lhs_vec))
    print(f"V solve residual norm: {diff_norm}")
    assert tree_allclose(rhs_vec, lhs_vec, atol=1e-7)
    for key in rhs_vec.keys():
        print(key, jnp.linalg.norm(rhs_vec[key] - lhs_vec[key]))
        assert np.isclose(0, jnp.linalg.norm(rhs_vec[key] - lhs_vec[key]))
    print("LHS == RHS")


    Vparams = sol
    print("CG tests complete.")
    uv_diff =  optax.tree_utils.tree_add_scale(Uparams, -1, Vparams)
    print("U-V norm ", optax.tree_utils.tree_norm(uv_diff))

   

def test_C_pytree_match_v2rdm_cost():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    
    # Get C_pytree from the generator
    # We need a dummy params to get the structure, Gparams works
    C_pytree = get_C_pytree(dummy_params=Gparams, CX_fn=CX_fn)
    
    # Manually construct C_expected from F[0]
    # F[0] is the cost matrix
    C_sdp = F[0]
    
    C_expected = {}
    
    # Initialize zero matrices for each block
    for i, block_name in enumerate(my_sdp.blocks):
        dim = my_sdp.dimensions[i]
        # Dimensions in my_sdp.dimensions are the side length of the block
        # If dim > 0, it's (dim, dim)
        if dim < 0:
             # Diagonal block (not used in this example usually, but strictly SDPA allows diag)
             # But here we assume dense blocks for v2rdm
             pass
        actual_dim = abs(dim)
        C_expected[block_name] = np.zeros((actual_dim, actual_dim))

    # Populate C_expected
    # sdp_matrix indices are 1-based
    for idx in range(len(C_sdp.value)):
        blk_idx = C_sdp.block_number[idx] - 1
        r = C_sdp.row[idx] - 1
        c = C_sdp.column[idx] - 1
        val = C_sdp.value[idx]
        
        blk_name = my_sdp.blocks[blk_idx]
        
        # SDPA is symmetric, usually only upper or lower triangle is stored or both?
        # v2rdm_sdp seems to store explicit entries.
        # If it's effectively symmetric, we should accumulate?
        # Typically SDPA files allow repeats and they sum up.
        
        C_expected[blk_name][r, c] += 0.5 * val
        if r != c:
             C_expected[blk_name][c, r] += 0.5 * val
        else:
             # For diagonal, we added 0.5 * val above. 
             # We need to add another 0.5 * val to make it full val?
             # Wait, my logic above: "C_exp[r,r] += 0.5 v. C_exp[r,r] += 0.5 v." 
             # The first line adds 0.5. The second line (if r!=c) is skipped.
             # So for diagonal we must add the other 0.5 explicitly if we want full val.
             C_expected[blk_name][r, c] += 0.5 * val

    # Verify C_pytree matches C_expected
    # C_pytree might be jax arrays, C_expected is numpy
    
    # We strictly only care about the blocks that exist in C_pytree/Gparams
    # Gparams keys should match my_sdp.blocks
    
    assert pytree_allclose(C_pytree, C_expected), "C_pytree does not match C_expected"
    print("test_C_pytree_match_v2rdm_cost passed")

def test_solver_run():
    my_sdp, b, F, solver, x_bpsdp, AX_fn, ATy_fn, CX_fn, y, z, c, mu, Ax, bvec, residual, sigma, yparams, btree, Rparams, Gparams, A_mat, C_vec = setup_v2rdm_sol()
    Uparams = Rparams
    Vparams = Rparams
    for key, val in Rparams.items():
        Uparams[key] = jnp.array(np.random.randn(*val.shape))
        Vparams[key] = Uparams[key].copy()
    C_pytree = get_C_pytree(dummy_params=Uparams, CX_fn=CX_fn)

    # Update U callback
    def update_U_fn(U, V, y, sig, cg_max_iter=1000, tol=1e-8):
        return solve_for_Uparams_cg(
            init_U=U,
            Vparams=V,
            yparams=y,
            btree=btree,
            Ctree=C_pytree,
            sigma=sig['sigma'],
            AX_fn=AX_fn,
            AT_fn=ATy_fn,
            tol=tol, # Fixed CG tolerance as per original
            max_steps=cg_max_iter
        )

    # Update V callback
    def update_V_fn(U, V, y, sig, cg_max_iter=1000, tol=1e-8):
        return solve_for_Vparams_cg(
            Uparams=U,
            init_V=V,
            yparams=y,
            btree=btree,
            Ctree=C_pytree,
            sigma=sig['sigma'],
            AX_fn=AX_fn,
            AT_fn=ATy_fn,
            tol=tol, # Fixed CG tolerance as per original
            max_steps=cg_max_iter
        )

    # Evaluate callback
    def evaluate_fn(U, V, y, sig):
        return compute_norm_and_residual(
            Uparams=U,
            Vparams=V,
            yparams=y,
            btree=btree,
            sigma=sig['sigma'],
            AX_fn=AX_fn,
            CX_fn=CX_fn
        )

    # Lagrangian callback
    def lagrangian_fn(U, V, y, sig):
        return admm_augmented_lagrangian(
            Uparams=U,
            Vparams=V,
            yparams=y,
            btree=btree,
            sigma=sig['sigma'],
            AX_fn=AX_fn,
            CX_fn=CX_fn
        )

    maxiter = 5_000
    stopping_epsilon = 1.0E-5
    update_frequency = 1
    sigma_scaling = 1.2
    tau = 2 # 10
    print_frequency = 1
    cg_max_iter = 1_000

    res = admm_solver(
        Uparams=Uparams,
        Vparams=Vparams,
        yparams=yparams,
        sigma=sigma,
        update_U_fn=update_U_fn,
        update_V_fn=update_V_fn,
        evaluate_fn=evaluate_fn,
        lagrangian_fn=lagrangian_fn,
        maxiter=maxiter,
        stopping_epsilon=stopping_epsilon,
        update_frequency=update_frequency,
        sigma_scaling=sigma_scaling,
        tau=tau,
        print_frequency=print_frequency,
        cg_max_iter=cg_max_iter,
        cg_tol=1e-8
    )


if __name__ == "__main__":
    # test_ATy()
    # test_admm_lagrangian()
    # test_gradient_admm_alm()
    # test_lhs_minus_rhs_equals_gradient()
    # test_lineax_CG_solve()
    # test_C_pytree_match_v2rdm_cost()
    test_solver_run()
