"""
Driver for variational two-electron reduced-density matrix method. Integrals come from PySCF
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

import pyscf

def build_sdp(nalpha, nbeta, nmo, oei, tei):
    """ set up details of the SDP

    :param nalpha:       number of alpha electrons
    :param nbeta:        number of beta electrons
    :param nmo:          number of spatial molecular orbitals
    :param oei:          core Hamiltonian matrix
    :param tei:          two-electron repulsion integrals
    :return: b:          the constraint vector
    :return: F:          list of rows of constraint matrix in sparse format; 
                         note that F[0] is actually the vector defining the 
                         problem (contains the one- and two-electron integrals)
    :return: dimensions: list of dimensions of blocks of primal solution
    """

    # for a two-electron system, all we need are
    # D1a D1b D2ab 

    # block dimensions
    dimensions = []
    dimensions.append(nmo)     # D1a
    dimensions.append(nmo)     # D1a
    dimensions.append(nmo*nmo) # D2ab

    # number of blocks
    nblocks = len(dimensions)

    # F0 
    block_number=[]
    row=[]
    column=[]
    value=[]

    F = [libsdp.sdp_matrix()]

    for i in range (0,nmo):
        for j in range (0,nmo):
            block_number.append(1)
            row.append(i+1)
            column.append(j+1)
            value.append(oei[i][j])

    for i in range (0,nmo):
        for j in range (0,nmo):
            block_number.append(2)
            row.append(i+1)
            column.append(j+1)
            value.append(oei[i][j])

    for i in range (0,nmo):
        for j in range (0,nmo):
            ij = i * nmo + j
            for k in range (0,nmo):
                for l in range (0,nmo):
                    kl = k * nmo + l
                    block_number.append(3)
                    row.append(ij+1)
                    column.append(kl+1)
                    value.append(tei[i][k][j][l])

    count = 0
    F[count].block_number = block_number
    F[count].row          = row
    F[count].column       = column
    F[count].value        = value
    count += 1
    
    # constraints (F1, F2, ...)

    b = []

    # Tr(D1a)
    block_number=[]
    row=[]
    column=[]
    value=[]

    for i in range (0,nmo):
        block_number.append(1)
        row.append(i+1)
        column.append(i+1)
        value.append(1.0)

    F.append(libsdp.sdp_matrix())
    F[count].block_number = block_number
    F[count].row          = row
    F[count].column       = column
    F[count].value        = value

    b.append(nalpha)

    count += 1

    # Tr(D1b)
    block_number=[]
    row=[]
    column=[]
    value=[]

    for i in range (0,nmo):
        block_number.append(2)
        row.append(i+1)
        column.append(i+1)
        value.append(1.0)

    F.append(libsdp.sdp_matrix())
    F[count].block_number = block_number
    F[count].row          = row
    F[count].column       = column
    F[count].value        = value

    b.append(nbeta)

    count += 1

    # Tr(D2ab)
    block_number=[]
    row=[]
    column=[]
    value=[]

    for i in range (0,nmo):
        for j in range (0,nmo):
            ij = i * nmo + j
            block_number.append(3)
            row.append(ij+1)
            column.append(ij+1)
            value.append(1.0)

    F.append(libsdp.sdp_matrix())
    F[count].block_number = block_number
    F[count].row          = row
    F[count].column       = column
    F[count].value        = value

    b.append(nalpha*nbeta)

    count += 1

    # D2ab -> D1a
    for i in range (0,nmo):
        for j in range (0,nmo):

            block_number=[]
            row=[]
            column=[]
            value=[]

            for k in range (0,nmo):

                ik = i * nmo + k
                jk = j * nmo + k
                block_number.append(3)
                row.append(ik+1)
                column.append(jk+1)
                value.append(1.0)

            block_number.append(1)
            row.append(i+1)
            column.append(j+1)
            value.append(-nbeta)
    
            F.append(libsdp.sdp_matrix())
            F[count].block_number = block_number
            F[count].row          = row
            F[count].column       = column
            F[count].value        = value

            b.append(0.0)

            count += 1

    # D2ab -> D1b
    for i in range (0,nmo):
        for j in range (0,nmo):

            block_number=[]
            row=[]
            column=[]
            value=[]

            for k in range (0,nmo):

                ki = k * nmo + i
                kj = k * nmo + j
                block_number.append(3)
                row.append(ki+1)
                column.append(kj+1)
                value.append(1.0)

            block_number.append(2)
            row.append(i+1)
            column.append(j+1)
            value.append(-nalpha)
    
            F.append(libsdp.sdp_matrix())
            F[count].block_number = block_number
            F[count].row          = row
            F[count].column       = column
            F[count].value        = value

            b.append(0.0)

            count += 1

    return b, F, dimensions

def main():

    # build molecule
    mol = pyscf.M(
        atom='H 0 0 0; H 0 0 1.0',
        basis='sto-3g',
        symmetry=False)

    # run RHF
    mf = mol.RHF().run()

    # get mo coefficient matrix
    C = mf.mo_coeff

    # get two-electron integrals
    tei = mol.intor('int2e')

    # transform two-electron integrals to mo basis
    tei = np.einsum('uj,vi,wl,xk,uvwx',C,C,C,C,tei)

    # get core hamiltonian
    kinetic   = mol.intor('int1e_kin')
    potential = mol.intor('int1e_nuc')
    oei       = kinetic + potential

    # transform core hamiltonian to mo basis
    oei = np.einsum('uj,vi,uv',C,C,oei)

    # number of occupied orbitals
    occ = mf.mo_occ
    nele = int(sum(occ))
    nalpha = nele // 2
    nbeta  = nalpha

    # number of spatial orbitals
    nmo = int(mf.mo_coeff.shape[1])

    # build inputs for the SDP
    # 
    # min   x.c
    # s.t.  Ax = b
    #       x >= 0
    # 
    # b is the right-hand side of Ax = b
    # F contains c followed by the rows of A, in SDPA sparse matrix format
    # 
    b, F, dimensions = build_sdp(nalpha,nbeta,nmo,oei,tei)

    # set options
    options = libsdp.sdp_options()

    maxiter = 5000000

    options.sdp_algorithm             = options.SDPAlgorithm.RRSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-8
    options.sdp_objective_convergence = 1e-8
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    sdp = libsdp.sdp_solver(options)
    x = sdp.solve(b,F,dimensions,maxiter)

    # primal energy:
    objective = 0
    for i in range (0,len(F[0].block_number)):
        block  = F[0].block_number[i] - 1
        row    = F[0].row[i] - 1
        column = F[0].column[i] - 1
        value  = F[0].value[i]
        
        off = 0
        for j in range (0,block):
            off += dimensions[j] * dimensions[j]

        objective += x[off + row * dimensions[block] + column] * value

    print('')
    print('    * v2RDM electronic energy: %20.12f' % (objective))
    print('    * v2RDM total energy:      %20.12f' % (objective + mf.energy_nuc()))
    print('')

if __name__ == "__main__":
    main()

