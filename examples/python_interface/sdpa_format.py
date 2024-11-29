import numpy as np
import sys
import libsdp

from libsdp.sdpa_file_io import read_sdpa_problem
from libsdp.sdp_helper import sdp_solver

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
    options = libsdp.sdp_options()
    
    maxiter = 500000
    
    options.sdp_algorithm             = "bpsdp"
    options.procedure                 = "maximize"
    options.guess_type                = "zero"
    #options.guess_type                = "read"
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.cg_convergence            = 1e-12
    options.dynamic_cg_convergence    = False
    #options.penalty_parameter_scaling = 0.1
    #options.penalty_parameter         = 0.1
   
    # solve sdp (python) 

    # solve sdp (c++)
    sdp = sdp_solver(options, A, block_dim)
    x = sdp.solve(b, maxiter)
    c = np.array(sdp.get_c())

    # don't forget minus sign since SDPLIB problems are maximizations
    primal_objective_value = -np.dot(c, x)

    # action of A on x 
    Ax = np.array(sdp.get_Au(x))

    primal_error = Ax - b

    print('')
    print('    objective value (primal): %20.12f' % (primal_objective_value))
    print('    ||Ax - b||:               %20.12f' % (np.linalg.norm(primal_error)))
    print('')

    # for bpsdp, we can also play around with the dual solutions
    if options.sdp_algorithm.lower() == "bpsdp" :

        z = np.array(sdp.get_z())
        y = np.array(sdp.get_y())

        # don't forget minus sign since SDPLIB problems are maximizations
        dual_objective_value = -np.dot(b, y)

        # action of A^T on y
        ATy = np.array(sdp.get_ATu(y))

        dual_error = c - z - ATy

        print('    objective value (dual):   %20.12f' % (dual_objective_value))
        print('    ||c - ATy - z||:          %20.12f' % (np.linalg.norm(dual_error)))
        print('    |c.x - b.y|:              %20.12f' % (np.linalg.norm(dual_objective_value - primal_objective_value)))
        print('')

if __name__ == "__main__":
    main()
