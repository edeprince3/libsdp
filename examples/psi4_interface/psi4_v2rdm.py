"""
Driver for variational two-electron reduced-density matrix method. Integrals come from Psi4
"""
import numpy as np
from numpy import einsum

import sys
sys.path.insert(0, '../../.')

import libsdp
from v2rdm_sdp import v2rdm_sdp
from g2_v2rdm_sdp import g2_v2rdm_sdp

import psi4

def main():

    # set molecule
    mol = psi4.geometry("""
    0 1
         b 0.0 0.0 0.0
         h 0.0 0.0 1.0
    no_reorient
    nocom
    symmetry c1
    """)

    # set options
    psi4_options_dict = {
        'basis': 'sto-3g',
        'reference': 'rohf',
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
    #my_sdp = v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, q2 = False, constrain_spin = False, g2 = True)
    my_sdp = g2_v2rdm_sdp(nalpha, nbeta, nmo, oei, tei, d2 = False, q2 = False, constrain_spin = False)

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

    # get individual constraint matrices and build up SOS hamiltonian
    a0_y = my_sdp.get_constraint_matrix(0) * y[0]
    print(a0_y, my_sdp.get_constraint_matrix(0), y[0])
    energy = 0.0
    ai_y = np.zeros(len(x), dtype = 'float64')
    for i in range (1, len(y)):
        a = my_sdp.get_constraint_matrix(i)
        ai_y = ai_y + a * y[i]

    # check that individual constraint matrices sum up correctly
    assert(np.isclose(0.0, np.linalg.norm(ATy - a0_y - ai_y)))

    # check that individual constraint matrices sum up correctly, again
    assert(np.isclose(np.linalg.norm(dual_error), np.linalg.norm(c - z - a0_y - ai_y)))

    # sum of squares hamiltonian
    c_sos = z + ai_y

    # check that c_sos . x = 0 ... this should approach zero with sufficiently tight convergence
    sos_energy = np.dot(c_sos, x)
    #print(sos_energy)

    # sum of squares hamiltonian, blocked
    c_sos = my_sdp.get_rdm_blocks(c_sos) 

    #import scipy
    #print('eigenvalues of SOS hamiltonian')
    #for i in range (0, len(c_sos)):
    #    print(c_sos[i].shape)
    #    w = scipy.linalg.eigh(c_sos[i], eigvals_only=True)
    #    print()
    #    print(w)

if __name__ == "__main__":
    main()

