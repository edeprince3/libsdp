# test python implementation to libsdp

import sys
sys.path.insert(0, '../.')

import libsdp

# define SDPOptions object
options = libsdp.sdp_options()

# number of primal variables (what SDPA considers to be dual variables)
n_primal = 8

# number of dual variables (what SDPA considers to be primal variables)
n_dual = 2

sdp = libsdp.sdp_solver(n_primal,n_dual,options)

# F0
F0 = libsdp.sdp_matrix()

F0.block_number  = [1,1,2,2]
F0.row           = [1,2,1,2] 
F0.column        = [1,2,1,2] 
F0.value         = [1.0,2.0,3.0,4.0]

# Fi

Fi = [libsdp.sdp_matrix()]

Fi[0].block_number = [1,1]
Fi[0].row          = [1,2]
Fi[0].column       = [1,2]
Fi[0].value        = [1.0,1.0]

Fi.append(libsdp.sdp_matrix())

Fi[1].block_number = [1,2,2,2,2]
Fi[1].row          = [2,1,1,2,2]
Fi[1].column       = [2,1,2,1,2]
Fi[1].value        = [1.0,5.0,2.0,2.0,6.0]

c = [10.0, 20.0]

block_dim = [2, 2]
maxiter = 50000

sdp.solve(c,F0,Fi,block_dim,maxiter)

#sdp.solve(x,b,c,primal_dim_block,maxiter,evaluate_Au,evaluate_ATu,progress_monitor)

