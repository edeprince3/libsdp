"""
Driver for variational two-electron reduced-density matrix method. Integrals come from PySCF
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp
from v2rdm_sdp import v2rdm_sdp

import pyscf

def main():

    # build molecule
    mol = pyscf.M(
        atom='B 0 0 0; H 0 0 1.0',
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
    my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei)

    b = my_sdp.b
    F = my_sdp.F
    dimensions = my_sdp.dimensions

    # set options
    options = libsdp.sdp_options()

    maxiter = 100000

    options.sdp_algorithm             = options.SDPAlgorithm.BPSDP
    options.maxiter                   = maxiter
    options.sdp_error_convergence     = 1e-4
    options.sdp_objective_convergence = 1e-4
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    sdp_solver = libsdp.sdp_solver(options)
    x = sdp_solver.solve(b, F, dimensions, maxiter)

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

