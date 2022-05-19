"""
Driver for spin-orbital v2RDM. Integrals come from psi4.
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp

import psi4

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

    # number of abeta electrons
    nbeta = wfn.nbeta()

    # total number of orbitals
    nmo     = wfn.nmo()
    
    # molecular orbitals (spatial):
    C = wfn.Ca()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the one-electron integrals
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    H = np.einsum('uj,vi,uv', C, C, H)

    # build the two-electron integrals:
    tei = np.asarray(mints.mo_eri(C, C, C, C))
    
    # SDP time

    # for a two-electron system, all we need are
    # D1a D1b D2ab 

    # constraints:
    # Tr(D1a)  = Na
    # Tr(D1a)  = Nb
    # Tr(D2ab) = Na Nb
    # D2ab -> D1a
    # D2ab -> D1b

    n_dual = 3 + 2 * nmo*nmo

    # block dimensions
    dimensions = []
    dimensions.append(nmo)
    dimensions.append(nmo)
    dimensions.append(nmo*nmo)

    # number of blocks
    nblocks = len(dimensions)

    n_primal = 2*nmo*nmo + nmo*nmo*nmo*nmo

    # F0 
    block_number=[]
    row=[]
    column=[]
    value=[]

    Fi = [libsdp.sdp_matrix()]

    for i in range (0,nmo):
        for j in range (0,nmo):
            block_number.append(1)
            row.append(i+1)
            column.append(j+1)
            value.append(-H[i][j])

    for i in range (0,nmo):
        for j in range (0,nmo):
            block_number.append(2)
            row.append(i+1)
            column.append(j+1)
            value.append(-H[i][j])

    for i in range (0,nmo):
        for j in range (0,nmo):
            ij = i * nmo + j
            for k in range (0,nmo):
                for l in range (0,nmo):
                    kl = k * nmo + l
                    block_number.append(3)
                    row.append(ij+1)
                    column.append(kl+1)
                    value.append(-tei[i][k][j][l])

    count = 0
    Fi[count].block_number = block_number
    Fi[count].row          = row
    Fi[count].column       = column
    Fi[count].value        = value
    count += 1
    
    # constraints (F1, F2, ...)

    c = []

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

    Fi.append(libsdp.sdp_matrix())
    Fi[count].block_number = block_number
    Fi[count].row          = row
    Fi[count].column       = column
    Fi[count].value        = value

    c.append(nalpha)

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

    Fi.append(libsdp.sdp_matrix())
    Fi[count].block_number = block_number
    Fi[count].row          = row
    Fi[count].column       = column
    Fi[count].value        = value

    c.append(nbeta)

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

    Fi.append(libsdp.sdp_matrix())
    Fi[count].block_number = block_number
    Fi[count].row          = row
    Fi[count].column       = column
    Fi[count].value        = value

    c.append(nalpha*nbeta)

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
    
            Fi.append(libsdp.sdp_matrix())
            Fi[count].block_number = block_number
            Fi[count].row          = row
            Fi[count].column       = column
            Fi[count].value        = value

            c.append(0.0)

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
    
            Fi.append(libsdp.sdp_matrix())
            Fi[count].block_number = block_number
            Fi[count].row          = row
            Fi[count].column       = column
            Fi[count].value        = value

            c.append(0.0)

            count += 1

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
    sdp.solve(c,Fi,dimensions,maxiter)

if __name__ == "__main__":
    main()

