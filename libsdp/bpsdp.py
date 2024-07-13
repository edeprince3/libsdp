import numpy as np
import sys
import libsdp

def bpsdp(x, b, c, primal_block_dim, Ai, options):
    """
    solve an SDP by the boundary-point algorithm
   
    :param x: the primal solution vector 
    :param b: the constraint vector
    :param c: the objective function vector
    :param primal_block_dim: list of block dimensions for the primal solution
    :param Ai: list of constraint matrices
    :param options: libsdp options object

    :return x: the primal solution vector
    :return y: the dual solution vector
    :return z: c - ATy

    """

    print('')
    print('    ==> BPSDP: Boundary-point SDP <==')
    print('')

    # number of primal variables
    n_primal = 0
    for i in range (0, len(primal_block_dim)):
        n_primal += int(primal_block_dim[i]**2)
 
    # number of constraints
    n_dual = len(Ai)

    y = np.zeros([n_dual])
    z = np.zeros([n_primal])

    mu = 0.1 
    tau = 1.0

    bpsdp_iter = 0

    print('      oiter iiter         c.x         b.y    |c.x-b.y|      mu    ||Ax-b|| ||ATy-c+z||')
    while bpsdp_iter < options.maxiter:

        # step 1: solve for y using the conjuage gradient method

        # evaluate A(c-z) + tau * mu * (b - Ax) for conjugate gradient
        cg_rhs = evaluate_Au(Ai, c-z) + tau * mu * (b - evaluate_Au(Ai, x))

        # solve AATy = A(c-z) + tau * mu * (b - Ax) via conjugate gradient
        cg_convergence = 1e-8
        if bpsdp_iter == 0:
            cg_convergence = 0.01
        else:
            if primal_error > dual_error :
                cg_convergence = 0.01 * dual_error
            else:
                cg_convergence = 0.01 * primal_error
        if cg_convergence < 1e-6:
            cg_convergence = 1e-6

        y, cg_iter = cg(y, cg_rhs, Ai, n_primal, cg_convergence)

        # step 2: update x and z
        
        U = mu * x + evaluate_ATu(Ai, y, n_primal) - c

        # reshape U and diagonalize each block
        off = 0
        for i in range (0, len(primal_block_dim)):
            n = primal_block_dim[i]
            Umat = U[off:off+n*n].reshape(n, n)
            Umat = 0.5 * (Umat + Umat.transpose(1,0))
            eigenvalues, eigenvectors = np.linalg.eigh(Umat)

            # split out positive / negative eigenvalues
            positive = np.diag(np.where(eigenvalues >= 0.0, eigenvalues, 0.0))
            negative = np.diag(np.where(eigenvalues < 0.0, eigenvalues, 0.0))

            # back transform and set x = U(+)/mu, z = -U(-)
            tmp = np.matmul(positive, eigenvectors.transpose())
            positive = np.matmul(eigenvectors, tmp)

            tmp = np.matmul(negative, eigenvectors.transpose())
            negative = np.matmul(eigenvectors, tmp)

            x[off:off + n*n] = positive.reshape(n*n) / mu
            z[off:off + n*n] = -negative.reshape(n*n)

            off += n*n

        # step 3: update mu

        # evaluate dual error || A^T y - c + z||
        dual_error = np.linalg.norm( evaluate_ATu(Ai, y, n_primal) - c + z)

        # evaluate primal error || Ax - b ||
        primal_error = np.linalg.norm( evaluate_Au(Ai, x) - b)

        # evaluate current primal and dual energies
        objective_primal = np.dot(c, x)
        objective_dual = np.dot(b, y)

        primal_dual_objective_gap = np.abs(objective_primal-objective_dual)

        # don't update mu every iteration
        if bpsdp_iter % 500 == 0 and bpsdp_iter > 0 :
            mu = mu * primal_error / dual_error

        print('      %5i %5i %11.6f %11.6f %11.6e %7.3f %10.5e %10.5e' % (bpsdp_iter, cg_iter, objective_primal, objective_dual, primal_dual_objective_gap, mu, primal_error, dual_error))

        if primal_error < options.sdp_error_convergence \
            and dual_error < options.sdp_error_convergence \
            and primal_dual_objective_gap < options.sdp_objective_convergence :
                break

        bpsdp_iter += 1

    if bpsdp_iter == options.maxiter :
        print('')
        print('    error: bpsdp did not converge')
        print('')
        exit()

    return x, y, z

def evaluate_Au(Ai, u):
    """
    evaluate action of constraint matrices on vector of dimension of primal vector

    :param Ai: list of constraint matrices
    :param u: vector of dimension of the primal vector
    :return Au: action of constraint matrices Ai on input vector u
    """

    Au = np.zeros([len(Ai)])
    for i in range (0, len(Ai)):
        dum = 0.0
        for j in range (0, Ai[i].value.size()):
            dum += Ai[i].value[j] * u[Ai[i].id[j]]
        Au[i] = dum

    return Au

def evaluate_ATu(Ai, u, n_primal):
    """
    evaluate action of transpose of constraint matrices on vector of dimension of dual vector

    :param Ai: list of constraint matrices
    :param u: vector of dimension of the dual vector
    :param n_primal: the number of primal variables
    :return ATu: action of transpose of constraint matrices Ai on input vector u
    """

    ATu = np.zeros([n_primal])
    for i in range (0, len(Ai)):
        for j in range (0, Ai[i].value.size()):
            ATu[Ai[i].id[j]] += Ai[i].value[j] * u[i]

    return ATu

def evaluate_AATu(Ai, u, n_primal):
    """
    evaluatef A.AT.u

    :param Ai: list of constraint matrices
    :param u: vector of dimension of the primal vector
    :param n_primal: the number of primal variables
    :return AATu
    """

    ATu = evaluate_ATu(Ai, u, n_primal)
    AATu = evaluate_Au(Ai, ATu)

    return AATu

def cg(y, rhs, Ai, n_primal, cg_convergence):
    """
    solve AATy = A(c-z) + tau * mu * (b - Ax) via conjugate gradient

    :param y: the dual solution vector
    :param rhs: the right-hand-side of the CG problem, tau * mu * (b - Ax)
    :param Ai: list of constraint matrices
    :param n_primal: the number of primal variables
    :return y
    """

    alpha = 0.0
    beta = 0.0

    r = rhs - evaluate_AATu(Ai, y, n_primal)
    p = r.copy()

    cg_iter = 0
    cg_maxiter = 1000
    
    while(cg_iter < cg_maxiter) :

        Ap = evaluate_AATu(Ai, p, n_primal)

        rr = np.dot(r, r)
        alpha = rr / np.dot(p, Ap)

        y += alpha * p
        r -= alpha * Ap

        rrnew = np.dot(r, r)
        nrm = np.sqrt(rrnew)
        if nrm < cg_convergence :
            break

        beta = rrnew / rr
        p = beta * p + r

        cg_iter += 1

    if cg_iter == cg_maxiter :
        print('')
        print('    error: maximum number of cg iterations exceeded')
        print('')
        exit()

    return y, cg_iter

def solve(b, Ai, primal_block_dim, options):

    n_primal = 0
    for i in range (0, len(primal_block_dim)):
        n_primal += int(primal_block_dim[i]**2)

    n_dual = len(Ai) - 1

    c = np.zeros([n_primal])
    x = 1e-4 * (np.random.rand(n_primal) - 0.5 )

    for i in range (0, Ai[0].block_number.size()):

        my_block  = Ai[0].block_number[i] - 1
        my_row = Ai[0].row[i] - 1
        my_column = Ai[0].column[i] - 1

        off = 0
        for j in range (0, my_block):
            off += int(primal_block_dim[j]**2)

        # populate relevant entry in c. 
        # note that, compared to SDPLIB problems, our definition of the problem 
        # has c = -F0. (we're minimizing; they maximize). the result will be that
        # our final objective will have the opposite sign compared to tabulated
        # SDPLIB values
        c[off + my_row * primal_block_dim[my_block] + my_column] += Ai[0].value[i]

    # constraint matrices
    my_Ai = []
    for i in range (1, len(Ai)):

        F = libsdp.sdp_matrix()

        # add composite indices
        for j in range (0, Ai[i].block_number.size()):

            # don't forget that the input Ai used unit-offset labels
            my_block = Ai[i].block_number[j] - 1
            my_row = Ai[i].row[j] - 1
            my_column = Ai[i].column[j] - 1

            F.block_number.append(my_block)
            F.row.append(my_row)
            F.column.append(my_column)
            F.value.append(Ai[i].value[j])

            # calculate offset
            off = 0
            for k in range (0, my_block):
                off += int(primal_block_dim[k]**2)

            # composite index
            idx = off + my_row * primal_block_dim[my_block] + my_column

            # add to matrix object
            F.id.append(idx)

        my_Ai.append(F)

    # call bpsdp
    bpsdp(x, b, c, primal_block_dim, my_Ai, options)

