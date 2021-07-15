# test python implementation to libsdp

import sys
sys.path.insert(0, '../.')

import libsdp

# let's solve this problem from the sdplib paper:

# "A sample problem
# 2 =mdim
# 2 =nblocks
# {2, 2}
# 10.0 20.0
# 0 1 1 1 1.0
# 0 1 2 2 2.0
# 0 2 1 1 3.0
# 0 2 2 2 4.0
# 1 1 1 1 1.0
# 1 1 2 2 1.0
# 2 1 2 2 1.0
# 2 2 1 1 5.0
# 2 2 1 2 2.0
# 2 2 2 2 6.0

# define SDPOptions object
options = libsdp.sdp_options()

# set the SDP solver algorithm (if you want ... default is BPSDP)
options.sdp_algorithm = options.SDPAlgorithm.RRSDP

# initialize sdp solver
sdp = libsdp.sdp_solver(options)

# F0 ... define as a single sdp_matrix object
F0 = libsdp.sdp_matrix()

block_number = []
row = []
column = []
value = []

# 0 1 1 1 1.0
block_number.append(1)
row.append(1)
column.append(1)
value.append(1.0)

# 0 1 2 2 2.0
block_number.append(1)
row.append(2)
column.append(2)
value.append(2.0)

# 0 2 1 1 3.0
block_number.append(2)
row.append(1)
column.append(1)
value.append(3.0)

# 0 2 2 2 4.0
block_number.append(2)
row.append(2)
column.append(2)
value.append(4.0)

F0.block_number = block_number
F0.row          = row
F0.column       = column
F0.value        = value

# Fi ... define as a list of sdp_matrix objects

# first object in Fi list (Fi[0] = F1 from input)
Fi = [libsdp.sdp_matrix()]

block_number = []
row = []
column = []
value = []

# 1 1 1 1 1.0
block_number.append(1)
row.append(1)
column.append(1)
value.append(1.0)

# 1 1 2 2 1.0
block_number.append(1)
row.append(2)
column.append(2)
value.append(1.0)

Fi[0].block_number = block_number
Fi[0].row          = row
Fi[0].column       = column
Fi[0].value        = value

# second object in Fi list (Fi[1] = F2 from input)
block_number = []
row = []
column = []
value = []

# 2 1 2 2 1.0
block_number.append(1)
row.append(2)
column.append(2)
value.append(1.0)

# 2 2 1 1 5.0
block_number.append(2)
row.append(1)
column.append(1)
value.append(5.0)

# 2 2 1 2 2.0
block_number.append(2)
row.append(1)
column.append(2)
value.append(2.0)

# 2 2 2 1 2.0
block_number.append(2)
row.append(2)
column.append(1)
value.append(2.0)

# 2 2 2 2 6.0
block_number.append(2)
row.append(2)
column.append(2)
value.append(6.0)

Fi.append(libsdp.sdp_matrix())

Fi[1].block_number = block_number
Fi[1].row          = row
Fi[1].column       = column
Fi[1].value        = value

# SDPA c vector (the constraint vector b, in our convention)
c = [10.0, 20.0]

# list of block dimensions
block_dim = [2, 2]

# maximum number of iterations
maxiter = 50000

# call the SDP solver
sdp.solve(c,F0,Fi,block_dim,maxiter)

