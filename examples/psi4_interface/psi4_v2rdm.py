"""
Driver for variational two-electron reduced-density matrix method. Integrals come from Psi4
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

import psi4

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

    # set molecule
    mol = psi4.geometry("""
    0 1
         H 0.0 0.0 0.0
         H 0.0 0.0 1.0
    no_reorient
    nocom
    symmetry c1
    """)

    # set options
    psi4_options_dict = {
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }
    psi4.set_options(psi4_options_dict)

    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # number of alpha electrons
    nalpha = wfn.nalpha()

    # number of beta electrons
    nbeta = wfn.nbeta()

    # total number of orbitals
    nmo     = wfn.nmo()

    # molecular orbitals (spatial):
    C = wfn.Ca()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the one-electron integrals
    oei = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    oei = np.einsum('uj,vi,uv', C, C, oei)

    # build the two-electron integrals:
    tei = np.asarray(mints.mo_eri(C, C, C, C))

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
    print('    * v2RDM total energy:      %20.12f' % (objective + mol.nuclear_repulsion_energy()))
    print('')

if __name__ == "__main__":
    main()

