"""
Driver for variational two-electron reduced-density matrix method. Integrals come from PySCF
"""
import numpy as np
from numpy import einsum

import sys

from v2rdm_sdp import v2rdm_sdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

from libsdp import sdp_options
from libsdp.sdp_helper import sdp_solver
from libsdp.sdpa_file_io import clean_sdpa_problem
from libsdp.sdpa_file_io import read_sdpa_problem
from libsdp.sdpa_file_io import write_sdpa_problem

import pyscf

def main():

    # build molecule
    # build molecule
    mol = pyscf.M(
        atom=[
['H', (0, 0, 0)],
['H', (0, 0, 1)],
['H', (0, 0, 2)],
['H', (0, 0, 3)],
['H', (0, 0, 4)],
['H', (0, 0, 5)],
              ],
        basis='sto-3g',
        spin=1,
        charge=+1,
        symmetry=False)

    # run RHF
    mf = mol.RHF().run()

    # get mo coefficient matrix
    C = mf.mo_coeff

    # get two-electron integrals
    tei = mol.intor('int2e')

    # transform two-electron integrals to mo basis
    tei = np.einsum('uj,vi,wl,xk,uvwx', C, C, C, C, tei)

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

    # use this one for d-only
    my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2 = True, g2 = True, constrain_spin = False)

    #b = my_sdp.b
    #F = my_sdp.F

    b, F = clean_sdpa_problem(my_sdp.b, my_sdp.F)

    dimensions = my_sdp.dimensions

    # set options
    options = sdp_options()


    #options.sdp_algorithm             = "bpsdp"
    #options.maxiter                   = maxiter
    #options.sdp_error_convergence     = 1e-4
    #options.sdp_objective_convergence = 1e-4
    #options.penalty_parameter_scaling = 0.1

    options.sdp_algorithm             = "bpsdp"
    #options.procedure                 = "maximize"
    options.guess_type                = "random"
    options.sdp_error_convergence     = 5e-5
    options.sdp_objective_convergence = 5e-5
    options.cg_convergence            = 1e-8
    options.dynamic_cg_convergence    = False

    # solve sdp
    maxiter = 100000
    sdp = sdp_solver(options, F, dimensions)
    x = sdp.solve(b, maxiter)

    # now that the sdp is solved, we can play around with the primal and dual solutions
    z = np.array(sdp.get_z())
    c = np.array(sdp.get_c())
    y = np.array(sdp.get_y())

    dual_energy = np.dot(b, y)
    primal_energy = np.dot(c, x)

    #dum = 0.0
    #for i in range (0, len(b)):
    #    if np.abs(b[i]) > 1e-6:
    #        dum += b[i] * y[i]
    #        print(b[i], y[i])
    #print()
    #print(dum)

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
    #ATy = my_sdp.get_rdm_blocks(ATy)

    print('')
    print('    * v2RDM electronic energy: %20.12f' % (primal_energy))
    print('    * v2RDM total energy:      %20.12f' % (primal_energy + mf.energy_nuc()))
    print('')
    print('    ||Ax - b||:                %20.12f' % (np.linalg.norm(primal_error)))
    print('    ||c - ATy - z||:           %20.12f' % (np.linalg.norm(dual_error)))
    print('    |c.x - b.y|:               %20.12f' % (np.linalg.norm(dual_energy - primal_energy)))
    print('')



    

if __name__ == "__main__":
    main()

