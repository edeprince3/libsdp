import numpy as np
import sys

from libsdp import sdp_options
from libsdp.sdpa_file_io import read_sdpa_problem

import libsdp.bpsdp 

def main():

    """
    Example for using libsdp with SDPA sparse format inputs

    max  tr (F0.y)
    s.t. tr (Fi.y) = ci, i = 1..m
    and  y \succeq 0

    In libSDP, this problem maps onto 

    min  -tr (c.x) 
    s.t. tr (Ai.x) = bi, i = 1..m
    and  x \suceeq 0

    Note the minus sign to account for maximize versus minimize.

    """

    filename = 'truss1.dat-s'

    # get constraint matrices (A) and constraint vector (b)
    # note that the first matrix in A (A0 or c) defines the objective function (A0.x = c.x)
    b, A, block_dim = read_sdpa_problem(filename)

    # set options
    options = sdp_options()
    
    
    options.sdp_algorithm             = "bpsdp"
    options.procedure                 = "maximize"
    options.guess_type                = "zero"
    #options.guess_type                = "read"
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.lbfgs_maxiter             = 50000 # for RRSDP
    options.cg_convergence            = 1e-12 # for BPSDP
    options.dynamic_cg_convergence    = False # for BPSDP
    #options.penalty_parameter_scaling = 0.1
    #options.penalty_parameter         = 0.1
   
    # solve sdp (python) 
    x, y, z, c = libsdp.bpsdp.solve(b, A, block_dim, options, maxiter = 50000)

    primal_objective_value = -np.dot(c, x)

    print('')
    print('    objective value (primal): %20.12f' % (primal_objective_value))
    print('')

if __name__ == "__main__":
    main()
