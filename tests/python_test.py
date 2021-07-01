# test python implementation to libsdp

import sys
sys.path.insert(0, '../../.')
import libsdp


def evaluate_Au(Au, u, data):

    pass

def evaluate_ATu(ATu, u, data):

    pass

def progress_monitor(oiter, iiter, objective_primal, objective_dual, 
    mu, primal_error, dual_error, data):

    pass

# define SDPOptions object
options = libsdp.sdp_options()

#print(options.sdp_algorithm)

# number of primal variables
n_primal = 1

# number of dual variables
n_dual = 1

sdp = libsdp.sdp_solver(n_primal,n_dual,options)

x = [0]
b = [0]
c = [0]
primal_dim_block = [1]

sdp.solve(x,b,c,primal_dim_block,evaluate_Au,evaluate_ATu,progress_monitor)

