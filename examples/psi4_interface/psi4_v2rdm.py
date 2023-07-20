"""
Driver for variational two-electron reduced-density matrix method. Integrals come from Psi4
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

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
         h 0.0 0.0 0.0
         h 0.0 0.0 1.0
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
    my_sdp = g2_v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2 = False, constrain_spin = True)

    b = my_sdp.b
    F = my_sdp.F
    dimensions = my_sdp.dimensions

    # set options
    options = libsdp.sdp_options()

    maxiter = 5000000

    options.sdp_algorithm             = options.SDPAlgorithm.BPSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-4
    options.sdp_objective_convergence = 1e-4
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    sdp = libsdp.sdp_solver(options)
    x = sdp.solve(b,F,dimensions,maxiter)

    # now that the sdp is solved, we can play around with the primal and dual solutions
    z = np.array(sdp.get_z())
    c = np.array(sdp.get_c())
    y = np.array(sdp.get_y())

    dual_energy = np.dot(b, y)
    primal_energy = np.dot(c, x)

    # action of A^T on y
    ATy = np.array(sdp.get_ATu(y))

    # action of A on x 
    Ax = np.array(sdp.get_Au(x))

    dual_error = c - z - ATy
    primal_error = Ax - b

    # extract blocks of rdms
    #x = my_sdp.get_rdm_blocks(x)
    #z = my_sdp.get_rdm_blocks(z)
    #c = my_sdp.get_rdm_blocks(c)

    #import scipy
    #print('eigenvalues')
    #for i in range (0, len(c)):
    #    wz = scipy.linalg.eigh(z[i], eigvals_only=True)
    #    wc = scipy.linalg.eigh(c[i], eigvals_only=True)
    #    print(wz, wc)

    print('')
    print('    * v2RDM electronic energy: %20.12f' % (primal_energy))
    print('    * v2RDM total energy:      %20.12f' % (primal_energy + mol.nuclear_repulsion_energy()))
    print('')
    print('    ||Ax - b||:                %20.12f' % (np.linalg.norm(primal_error)))
    print('    ||c - ATy - z||:           %20.12f' % (np.linalg.norm(dual_error)))
    print('    |c.x - b.y|:               %20.12f' % (np.linalg.norm(dual_energy - primal_energy)))
    print('')


if __name__ == "__main__":
    main()

