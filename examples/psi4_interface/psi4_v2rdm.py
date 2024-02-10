"""
Driver for variational two-electron reduced-density matrix method. Integrals come from Psi4
"""
import numpy as np
from numpy import einsum

import sys
import libsdp
from v2rdm_sdp import v2rdm_sdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

import psi4

def main():

    # set molecule
    mol = psi4.geometry("""
    0 1
         H
         H 1 1.0
    symmetry c1
    """)

    # set options
    psi4_options_dict = {
        'basis': '6-31g',
        'scf_type': 'pk',
    }
    psi4.set_options(psi4_options_dict)

    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # molecular orbitals (spatial):
    C = wfn.Ca()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the one-electron integrals
    # build the one-electron integrals
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    oei = np.einsum('uj,vi,uv', C, C, T + V)

    # build the two-electron integrals in the MO basis:
    tei = np.asarray(mints.mo_eri(C, C, C, C))

    # number of alpha electrons
    nalpha = wfn.nalpha()

    # number of beta electrons
    nbeta = wfn.nbeta()

    # total number of orbitals
    nmo     = wfn.nmo()



    # build inputs for the SDP
    # 
    # min   x.c
    # s.t.  Ax = b
    #       x >= 0
    # 
    # b is the right-hand side of Ax = b
    # F contains c followed by the rows of A, in SDPA sparse matrix format
    # 
    #my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2 = True, constrain_spin = True, g2 = True)
    my_sdp = g2_v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, d2 = True, q2 = True, constrain_spin = True)

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
    x = sdp.solve(b, F, dimensions, maxiter)

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
    #ATy = my_sdp.get_rdm_blocks(ATy)

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

