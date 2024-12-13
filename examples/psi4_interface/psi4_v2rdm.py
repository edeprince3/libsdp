"""
Driver for variational two-electron reduced-density matrix method. Integrals come from Psi4
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

import psi4

def main():

    # set molecule
    mol = psi4.geometry("""
    1 2
         H 0.0 0.0 0.0
         H 0.0 0.0 1.0
         H 0.0 0.0 2.0
         H 0.0 0.0 3.0
         H 0.0 0.0 4.0
         H 0.0 0.0 5.0
    symmetry c1
    """)

    # set options
    psi4_options_dict = {
        'reference': 'rohf',
        'basis': 'sto-3g',
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
    my_sdp = g2_v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, d2 = True, q2 = True, constrain_spin = False)

    dimensions = my_sdp.dimensions
    b, F = clean_sdpa_problem(my_sdp.b, my_sdp.F)

    # set options
    options = sdp_options()

    options.sdp_algorithm             = "bpsdp"
    options.sdp_error_convergence     = 1e-4
    options.sdp_objective_convergence = 1e-4
    options.penalty_parameter_scaling = 0.1

    # solve sdp
    maxiter = 5000000
    sdp = sdp_solver(options, F, dimensions)
    x = sdp.solve(b, maxiter)

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

