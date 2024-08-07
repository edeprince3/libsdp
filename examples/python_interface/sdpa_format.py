import numpy as np
import sys
import libsdp
import libsdp.bpsdp

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
            #for j in range (0,-my_dim):
            #    block_dim.append(1)
    
    # TODO: will "float" have the correct precision?
    c=(my_file[offset+3].split())
    for i in range(0,len(c)):
        c[i] = float((c[i]))
                     
    # Fi ... a list of sdp_matrix objects
    Fi = []
    
    current_block = 0

    F = libsdp.sdp_matrix()
    
    for i in range(offset+4,len(my_file)):
    
        temp=(my_file[i].split())
        my_block = int(temp[0])
    
        # whenever constraint number changes, we need to add the new matrix
        if ( my_block != current_block ) :
    
            current_block = my_block
    
            # append F to Fi
            Fi.append(F)
    
            # create new matrix object
            F = libsdp.sdp_matrix()
    
        # append constraint matrix values
        F.block_number.append(int(temp[1]))
        F.row.append(int(temp[2]))
        F.column.append(int(temp[3]))
        F.value.append(float(temp[4]))
    
        # remember, SDPA assumes constraint matrices are symmetric
        if ( int(temp[2]) != int(temp[3]) ):
            F.block_number.append(int(temp[1]))
            F.row.append(int(temp[3]))
            F.column.append(int(temp[2]))
            F.value.append(float(temp[4]))
    
    # assign last set of arrays in Fi
    Fi.append(F)

    return c, Fi, block_dim

def main():

    """
    Example for using libsdp with SDPA sparse format inputs
    """

    filename = 'c_example.in'
    #filename = 'SDPLIB/data/truss5.dat-s'
    #filename = 'truss1.dat-s'
    #filename = 'arch0.dat-s'
    #filename = 'SDPLIB/data/gpp100.dat-s'

    c, Fi, block_dim = read_sdp_problem(filename)

    # set options
    options = libsdp.sdp_options()
    
    maxiter = 5000000
    
    options.sdp_algorithm             = "bpsdp"
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-6
    options.sdp_objective_convergence = 1e-6
    options.penalty_parameter_scaling = 0.1
   
    # solve sdp (python) 
    #libsdp.bpsdp.solve(c, Fi, block_dim, options)

    # solve sdp (c++)
    sdp = libsdp.sdp_solver(options)
    sdp.solve(c,Fi,block_dim,maxiter)

if __name__ == "__main__":
    main()
