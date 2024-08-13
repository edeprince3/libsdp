import numpy as np
import sys
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
            #for j in range (0,-my_dim):
            #    block_dim.append(1)
   
    # TODO: will "float" have the correct precision?
    c=(my_file[offset+3].split())
    for i in range(0,len(c)):
        c[i] = float((c[i]))
                     
    # Fi ... define as a list of sdp_matrix objects
    Fi = []
    
    current_block = 0
    
    F = libsdp.sdp_matrix()
    
    for i in range(offset+4,len(my_file)):
    
        temp=(my_file[i].split())
        my_block = int(temp[0])
    
        # whenever constraint number changes, we need to add the new matrix
        if ( my_block != current_block ) :
    
            current_block = my_block
    
            # assign arrays in Fi
            Fi.append(F)
    
            # new sdp_matrix
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

    max  A0.x = c.x
    s.t. Ai.x = bi
         x >= 0

    """

    #filename = 'c_example.in'
    filename = 'truss1.dat-s'

    # get constraint matrices (A) and constraint vector (b)
    # note that the first matrix in A (A0 or c) defines the objective function (A0.x = c.x)
    b, A, block_dim = read_sdp_problem(filename)

    # set options
    options = libsdp.sdp_options()
    
    maxiter = 500000
    
    options.sdp_algorithm             = "bpsdp"
    options.procedure                 = "maximize"
    options.guess_type                = "random"
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.penalty_parameter_scaling = 0.1
    options.penalty_parameter         = 0.1
    options.cg_convergence            = 1e-12
   
    # solve sdp (python) 

    # solve sdp (c++)
    sdp = libsdp.sdp_solver(options)
    x = sdp.solve(b, A, block_dim, maxiter)
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
