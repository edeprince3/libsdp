import libsdp
import numpy as np

def read_sdpa_problem(filename):

    """
    Read a problem represented in SDPA sparse format.

    See http://euler.nmt.edu/~brian/sdplib/sdplib.pdf for additional
    details.

    The dual of the SDPA problem is similar to we call the primal problem
    in libSDP:

    max  tr (F0.y)
    s.t. tr (Fi.y) = ci, i = 1..m
    and  y \succeq 0

    In libSDP, this problem maps onto 

    min  -tr (c.x) 
    s.t. tr (Ai.x) = bi, i = 1..m
    and  x \suceeq 0

    Note the minus sign to account for maximize versus minimize.

    Note also that SDPA sparse format assumes that the constraint matrices
    are symmetric so only the upper triangle is given. This function adds
    the lower triangle terms.

    :param filename: the name of a file containing the problem,
    represented in SDPA sparse format

    :return c: the constraint vector (ci or bi above)

    :return Fi: a list of sdp_matrix objects defining the constraints. the
    first object in the  which (F0) defines the objective function

    :return block_dimensions: a list of dimensions for each block of the
    constraint matrices

    """

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
    tmp_block_dimensions=(my_file[offset+2].split())
    block_dimensions=[]
    for i in range(0,len(tmp_block_dimensions)):
        my_dim = int(tmp_block_dimensions[i])
        if my_dim > 0:
            block_dimensions.append(my_dim)
        else:
            block_dimensions.append(-my_dim)
            #for j in range (0,-my_dim):
            #    block_dimensions.append(1)
   
    c=(my_file[offset+3].split())
    for i in range(0,len(c)):
        c[i] = np.float64((c[i]))
                     
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

    return c, Fi, block_dimensions

def write_sdpa_problem(filename, c, Fi, block_dimensions):

    """
    Write a file containing a problem represented in SDPA sparse format.

    See http://euler.nmt.edu/~brian/sdplib/sdplib.pdf for additional
    details.

    The dual of the SDPA problem is similar to we call the primal problem
    in libSDP:

    max  tr (F0.y)
    s.t. tr (Fi.y) = ci, i = 1..m
    and  y \succeq 0

    In libSDP, this problem maps onto 

    min  -tr (c.x) 
    s.t. tr (Ai.x) = bi, i = 1..m
    and  x \suceeq 0

    Note the minus sign to account for maximize versus minimize.

    Note also that SDPA sparse format assumes that the constraint matrices
    are symmetric so only the upper triangle is given. As such, this
    function will only print the uppder triangle terms.

    :param filename: the name of the target file to which we will write
    the problem, represented in SDPA sparse format

    :param ci: the constraint vector (ci or bi above)

    :param Fi: a list of sdp_matrix objects defining the constraints. the
    first object in the  which (F0) defines the objective function

    :param block_dimensions: a list of dimensions for each block of the
    constraint matrices

    """

    f = open(filename, "w")

    # number of constraints:
    f.write("%10i\n" % (len(ci)))

    # number of blocks:
    f.write("%10i\n" % (len(block_dimensions)))

    # block dimensions
    s = ''
    for i in range (0, len(block_dimensions)):
        f.write("%10i " % (block_dimensions[i]))
    f.write("\n")

    # constraint values
    s = ''
    for i in range (0, len(ci)):
        f.write("%20.12e " % (ci[i]))
    f.write("\n")

    # Fi
    for i in range (0, len(Fi)):
        s = ''
        for j in range (0, Fi[i].block_number.size()):
            if Fi[i].row[j] <= Fi[i].column[j] : 
                f.write("%5i %5i %5i %5i %20.12e\n" % (i, Fi[i].block_number[j], Fi[i].row[j], Fi[i].column[j], Fi[i].value[j]) )

