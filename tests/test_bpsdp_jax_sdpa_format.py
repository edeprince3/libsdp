"""
Example for using libsdp with SDPA sparse format inputs. This example
uses the jax implementation of bpsdp, which runs on CPU or GPU.

max  tr (F0.y)
s.t. tr (Fi.y) = ci, i = 1..m
and  y succeq 0

In libSDP, this problem maps onto

min  -tr (c.x)
s.t. tr (Ai.x) = bi, i = 1..m
and  x succeq 0

Note the minus sign to account for maximize versus minimize.

"""

import os

import numpy as np
import scipy.sparse as sp
import pytest

pytest.importorskip("jax", reason="jax is not installed; skipping bpsdp_jax test")

from libsdp import sdp_options
from libsdp.sdpa_file_io import read_sdpa_problem

import libsdp.bpsdp_jax

def sdpa_to_sparse(A, block_dim, procedure):
    """
    convert the sdp_matrix list from read_sdpa_problem into the objective
    vector c and the sparse constraint matrix expected by bpsdp_jax.bpsdp

    :param A: list of sdp_matrix objects with unit-offset labels; A[0] defines
              the objective, A[1:] the constraints
    :param block_dim: list of block dimensions
    :param procedure: "minimize" or "maximize"

    :return c: the objective function vector
    :return Amat: the constraint matrix (scipy sparse)
    """

    dims = np.array(block_dim)
    offs = np.concatenate([[0], np.cumsum(dims * dims)])
    n_primal = int(offs[-1])

    def composite_index(F):
        block = np.array(F.block_number) - 1
        row = np.array(F.row) - 1
        column = np.array(F.column) - 1
        return offs[block] + row * dims[block] + column

    # objective. maximize c.x <-> minimize -c.x
    sg = -1.0 if procedure == "maximize" else 1.0
    c = np.zeros(n_primal)
    np.add.at(c, composite_index(A[0]), sg * np.array(A[0].value))

    # constraints, as one sparse matrix (duplicate entries are summed)
    rows = np.concatenate([np.full(len(F.value), i) for i, F in enumerate(A[1:])])
    cols = np.concatenate([composite_index(F) for F in A[1:]])
    vals = np.concatenate([np.array(F.value, dtype=float) for F in A[1:]])
    Amat = sp.csr_matrix((vals, (rows, cols)), shape=(len(A) - 1, n_primal))

    return c, Amat

def test_bpsdp_jax_sdpa_format():

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'truss1.dat-s')

    # get constraint matrices (A) and constraint vector (b)
    # note that the first matrix in A (A0 or c) defines the objective function (A0.x = c.x)
    b, A, block_dim = read_sdpa_problem(filename)

    # set options
    options = sdp_options()

    options.sdp_algorithm             = "bpsdp"
    options.procedure                 = "maximize"
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.cg_convergence            = 1e-12
    options.dynamic_cg_convergence    = False

    c, Amat = sdpa_to_sparse(A, block_dim, options.procedure)

    # zero guess
    x = np.zeros(len(c))

    # solve sdp (jax)
    x, y, z = libsdp.bpsdp_jax.bpsdp(x, np.array(b), c, block_dim, Amat, options, maxiter = 50000)

    primal_objective_value = -np.dot(c, x)

    print('')
    print('    objective value (primal): %20.12f' % (primal_objective_value))
    print('')

    assert np.isclose(primal_objective_value, -8.999996308753, atol=1e-8), "BPSDP primal objective value does not match reference value"

def main():
    raise Exception("run with pytest")

if __name__ == "__main__":
    main()
