# test python implementation to libsdp

import sys
sys.path.insert(0, '../.')

import libsdp

# define SDPOptions object
options = libsdp.sdp_options()

# number of primal variables (are these backward?)
n_primal = 2

# number of dual variables (are these backward?)
n_dual = 2

sdp = libsdp.sdp_solver(n_primal,n_dual,options)

x = [0.0]
b = [0.0]
c = [0.0]
primal_dim_block = [1]
maxiter = 50000

#sdp.solve(x,b,c,primal_dim_block,maxiter)
#sdp.solve(x,b,c,primal_dim_block,maxiter,evaluate_Au,evaluate_ATu,progress_monitor)

