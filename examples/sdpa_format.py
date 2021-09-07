import numpy as np

import sys
sys.path.insert(0, '../.')

import libsdp

def read_sdp_problem(filename):

    # input file
    h = open(filename, 'r')
     
    # read file
    my_file = h.readlines()
    
    # need to account for any comments at the top of the file
    offset = 0
    for i in range(0,len(my_file)):
        if ( my_file[i][0] == '"' or my_file[i][0] == '*' ):
            offset += 1
        else:
            break
    
    # number of constraints 
    m=int(my_file[offset])
    
    # number of blocks 
    nblocks=int(my_file[offset+1])
    
    # block dimensions 
    # TODO: need to account for formatting including {,}
    tmp_block_dim=(my_file[offset+2].split())
    block_dim=[]
    for i in range(0,len(tmp_block_dim)):
        my_dim = int(tmp_block_dim[i])
        if my_dim > 0:
            block_dim.append(my_dim)
        else:
            block_dim.append(-my_dim)
            #for j in range (0,-my_dim+1):
            #    block_dim.append(1)
    
    # TODO: will "float" have the correct precision?
    c=(my_file[offset+3].split())
    for i in range(0,len(c)):
        c[i] = float((c[i]))
                     
    # Fi ... define as a list of sdp_matrix objects
    Fi = [libsdp.sdp_matrix()]
    
    current_block = 0
    
    block_number=[]
    row=[]
    column=[]
    value=[]
    
    count = 0
    for i in range(offset+4,len(my_file)):
    
        temp=(my_file[i].split())
        my_block = int(temp[0])
    
        # whenever constraint number changes, we need to add the new matrix
        if ( my_block != current_block ) :
    
            current_block = my_block
    
            # assign arrays in Fi
            Fi[count].block_number = block_number
            Fi[count].row          = row
            Fi[count].column       = column
            Fi[count].value        = value
    
            # update number of constraint blocks
            count = count + 1
    
            # reset temporary arrays
            block_number=[]
            row=[]
            column=[]
            value=[]
    
            # create new matrix object
            Fi.append(libsdp.sdp_matrix())
    
        # append constraint matrix values
        block_number.append(int(temp[1]))
        row.append(int(temp[2]))
        column.append(int(temp[3]))
        value.append(float(temp[4]))
    
        # remember, SDPA assumes constraint matrices are symmetric
        if ( int(temp[2]) != int(temp[3]) ):
            block_number.append(int(temp[1]))
            row.append(int(temp[3]))
            column.append(int(temp[2]))
            value.append(float(temp[4]))
    
    # assign last set of arrays in Fi
    Fi[count].block_number = block_number
    Fi[count].row          = row
    Fi[count].column       = column
    Fi[count].value        = value

    return c, Fi, block_dim

def main():

    """
    Example for using libsdp with SDPA sparse format inputs
    """

    #filename = 'SDPLIB/data/truss5.dat-s'
    filename = 'truss1.dat-s'
    c, Fi, block_dim = read_sdp_problem(filename)
    
    # set options
    options = libsdp.sdp_options()
    
    maxiter = 5000000
    
    options.sdp_algorithm             = options.SDPAlgorithm.RRSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-7
    options.sdp_objective_convergence = 1e-6
    options.penalty_parameter_scaling = 0.5
    
    # solve sdp
    sdp = libsdp.sdp_solver(options)
    sdp.solve(c,Fi,block_dim,maxiter)

if __name__ == "__main__":
    main()
