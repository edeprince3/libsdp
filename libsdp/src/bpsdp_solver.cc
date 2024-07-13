/* 
 *  @BEGIN LICENSE
 * 
 *  libsdp: a library of semidefinite programming solvers
 * 
 *  Copyright (c) 2021 by its authors (LICENSE).
 * 
 *  The copyrights for code used from other parties are included in
 *  the corresponding files.
 * 
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * 
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 * 
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 * 
 *  @END LICENSE
 */

#include <string.h>
#include <memory>

#include <bpsdp_solver.h>
#include <cg_solver.h>

#include "blas_helper.h"

namespace libsdp {

// CG callback function
static void evaluate_cg_AATu(double * Ax, double * x, void * data) {

    // reinterpret void * as an instance of BPSDPSolver
    BPSDPSolver* sdp = reinterpret_cast<BPSDPSolver*>(data);
    sdp->evaluate_AATu(Ax, x);

}

void BPSDPSolver::evaluate_AATu(double * AATu,double * u) {

    evaluate_ATu_(ATu_,u,data_);
    evaluate_Au_(AATu,ATu_,data_);

}

BPSDPSolver::BPSDPSolver(long int n_primal, long int n_dual,SDPOptions options)
    : SDPSolver(n_primal,n_dual,options) {

    cg_rhs_ = (double*)malloc(n_dual_*sizeof(double));
    memset((void*)cg_rhs_,'\0',n_dual_*sizeof(double));

    ATu_ = (double*)malloc(n_primal_*sizeof(double));
    memset((void*)ATu_,'\0',n_primal_*sizeof(double));
}

BPSDPSolver::~BPSDPSolver(){
    free(cg_rhs_);
    free(ATu_);
}


void BPSDPSolver::solve(double * x,   
                        double * b, 
                        double * c,
                        std::vector<int> primal_block_dim,
                        int maxiter,
                        SDPCallbackFunction evaluate_Au, 
                        SDPCallbackFunction evaluate_ATu, 
                        SDPProgressMonitorFunction progress_monitor,
                        void * data){

    data_         = data;
    evaluate_Au_  = evaluate_Au;
    evaluate_ATu_ = evaluate_ATu;

    // cg solver
    std::shared_ptr<CGSolver> cg (new CGSolver(n_dual_));

    cg->set_max_iter(options_.cg_maxiter);
    cg->set_convergence(options_.cg_convergence);

    // the iterations
    double primal_dual_objective_gap = 0.0;

    int oiter_local = 0;

    do {

        // evaluate mu * (b - Ax) for CG
        evaluate_Au(Au_, x, data);
        C_DAXPY(n_dual_,-1.0,b,1,Au_,1);
        C_DSCAL(n_dual_,-mu_,Au_,1);

        // evaluate A(c-z) ( but don't overwrite c! )
        C_DSCAL(n_primal_,-1.0,z_,1);
        C_DAXPY(n_primal_,1.0,c,1,z_,1);
        evaluate_Au(cg_rhs_, z_, data);

        // add tau*mu*(b-Ax) to A(c-z) and put result in cg_rhs_
        C_DAXPY(n_dual_,1.0,Au_,1,cg_rhs_,1);

        // set convergence for CG problem (step 1 in table 1 of PRL 106 083001)
        double cg_conv_i = options_.cg_convergence;
        if (oiter_ == 0)
            cg_conv_i = 0.01;
        else
            cg_conv_i = (primal_error_ > dual_error_) ? 0.01 * dual_error_ : 0.01 * primal_error_;
        if (cg_conv_i < options_.cg_convergence)
            cg_conv_i = options_.cg_convergence;
        cg->set_convergence(cg_conv_i);

        // solve CG problem (step 1 in table 1 of PRL 106 083001)
        cg->solve(Au_,y_,cg_rhs_,evaluate_cg_AATu,(void*)this);
        int iiter = cg->total_iterations();

        iiter_total_ += iiter;

        // update primal and dual solutions
        Update_xz(x, c, primal_block_dim, evaluate_ATu, data);

        // update mu (step 3)

        // evaluate || A^T y - c + z||
        evaluate_ATu(ATu_, y_, data);
        C_DAXPY(n_primal_,1.0,z_,1,ATu_,1);
        C_DAXPY(n_primal_,-1.0,c,1,ATu_,1);
        dual_error_ = C_DNRM2(n_primal_,ATu_,1);

        // evaluate || Ax - b ||
        evaluate_Au(Au_, x, data);
        C_DAXPY(n_dual_,-1.0,b,1,Au_,1);
        primal_error_ = C_DNRM2(n_dual_,Au_,1);

        // compute current primal and dual energies
        double objective_primal = C_DDOT(n_primal_,c,1,x,1);
        double objective_dual   = C_DDOT(n_dual_,b,1,y_,1);

        primal_dual_objective_gap = fabs(objective_primal-objective_dual);

        progress_monitor(oiter_,iiter,objective_primal,objective_dual,mu_,primal_error_,dual_error_, data);

        oiter_++;
        oiter_local++;

        // don't update mu every iteration
        if ( oiter_ % options_.mu_update_frequency == 0 && oiter_ > 0 ){
            mu_ = mu_ * primal_error_ / dual_error_;
        }

        if ( primal_error_ > options_.sdp_error_convergence || dual_error_ > options_.sdp_error_convergence  || primal_dual_objective_gap > options_.sdp_objective_convergence ) {
            is_converged_ = false;
        }else {
            is_converged_ = true;
        }

        if ( oiter_local == maxiter ) break;
    }while( !is_converged_ );

}

// update x and z
void BPSDPSolver::Update_xz(double * x, double * c, std::vector<int> primal_block_dim, SDPCallbackFunction evaluate_ATu, void * data) {

    // evaluate M(mu*x + ATy - c)
    evaluate_ATu(ATu_, y_, data);
    C_DAXPY(n_primal_,-1.0,c,1,ATu_,1);
    C_DAXPY(n_primal_,mu_,x,1,ATu_,1);

    char char_n = 'n';
    char char_t = 't';
    double one = 1.0;
    double zero = 0.0;

    // loop over each block of x/z
    for (int i = 0; i < primal_block_dim.size(); i++) {
        if ( primal_block_dim[i] == 0 ) continue;
        int myoffset = 0;
        for (int j = 0; j < i; j++) {
            myoffset += primal_block_dim[j] * primal_block_dim[j];
        }

        long int block_dim = primal_block_dim[i];

        double * mat     = (double*)malloc(block_dim*block_dim*sizeof(double));
        double * eigvec  = (double*)malloc(block_dim*block_dim*sizeof(double));
        double * eigval  = (double*)malloc(block_dim*sizeof(double));

        for (int p = 0; p < block_dim; p++) {
            for (int q = p; q < block_dim; q++) {
                double dum = 0.5 * ( ATu_[myoffset + p * block_dim + q] +
                                     ATu_[myoffset + q * block_dim + p] );
                mat[p*block_dim + q] = mat[q*block_dim + p] = dum;
            }
        }

        Diagonalize(block_dim,mat,eigval);
        C_DCOPY(block_dim*block_dim,mat,1,eigvec,1);

        // separate U+ and U-, transform back to nondiagonal basis

        double * eigvec2 = (double*)malloc(block_dim*block_dim*sizeof(double));

        // (+) part
        long int mydim = 0;
        for (long int j = 0; j < block_dim; j++) {
            if ( eigval[j] > 0.0 ) {
                for (long int q = 0; q < block_dim; q++) {
                    mat[q*block_dim+mydim]     = eigvec[j*block_dim+q] * eigval[j]/mu_;
                    eigvec2[q*block_dim+mydim] = eigvec[j*block_dim+q];
                }
                mydim++;
            }
        }

        F_DGEMM(&char_t,&char_n,&block_dim,&block_dim,&mydim,&one,mat,&block_dim,eigvec2,&block_dim,&zero,&x[myoffset],&block_dim);

        // (-) part
        mydim = 0;
        for (long int j = 0; j < block_dim; j++) {
            if ( eigval[j] < 0.0 ) {
                for (long int q = 0; q < block_dim; q++) {
                    mat[q*block_dim+mydim]     = -eigvec[j*block_dim+q] * eigval[j];
                    eigvec2[q*block_dim+mydim] =  eigvec[j*block_dim+q];
                }
                mydim++;
            }
        }

        F_DGEMM(&char_t,&char_n,&block_dim,&block_dim,&mydim,&one,mat,&block_dim,eigvec2,&block_dim,&zero,&z_[myoffset],&block_dim);

        free(mat);
        free(eigvec);
        free(eigvec2);
        free(eigval);

    }

}

}

