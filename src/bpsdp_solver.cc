/* 
 *  @BEGIN LICENSE
 * 
 *  libsdp: a c++ library for solving semidefinite programs
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

#include "bpsdp_solver.h"
#include "blas_helper.h"
#include "cg_solver.h"
#include "omp.h"

namespace libsdp {

// CG callback function
static void evaluate_cg_AATu(double * Ax, double * x, void * data) {

    // reinterpret void * as an instance of BPSDPSolver
    BPSDPSolver* sdp = reinterpret_cast<BPSDPSolver*>(data);
    sdp->evaluate_AATu(Ax, x);

}

void BPSDPSolver::evaluate_AATu(double * AATu,double * u) {

    memset((void*)AATu,'\0',n_dual_*sizeof(double));

    double * ATu = (double*)malloc(n_primal_*sizeof(double));
    memset((void*)ATu,'\0',n_primal_*sizeof(double));

    evaluate_ATu_(ATu,u,data_);
    evaluate_Au_(AATu,ATu,data_);

    free(ATu);
}

BPSDPSolver::BPSDPSolver(long int n_primal, long int n_dual)
    : SDPSolver(n_primal,n_dual) {

    cg_rhs_ = (double*)malloc(n_dual_*sizeof(double));
    memset((void*)cg_rhs_,'\0',n_dual_*sizeof(double));
}

BPSDPSolver::~BPSDPSolver(){
    free(cg_rhs_);
}


void BPSDPSolver::solve(double * x,   
                        double * b, 
                        double * c,
                        std::vector<int> primal_block_dim,
                        int maxiter,
                        SDPCallbackFunction evaluate_Au, 
                        SDPCallbackFunction evaluate_ATu, 
                        void * data){

    data_         = data;
    evaluate_Au_  = evaluate_Au;
    evaluate_ATu_ = evaluate_ATu;

    // cg solver
    std::shared_ptr<CGSolver> cg (new CGSolver(n_dual_));
    // TODO: get CG_MAXITER from set() function
    cg->set_max_iter(10000);
    // TODO: get CG_CONVERGENCE from set() function
    double cg_convergence = 1e-8;
    cg->set_convergence(cg_convergence);

    // the iterations
    printf("\n");
    printf("    initial primal energy: %20.12lf\n",C_DDOT(n_primal_,c,1,x,1));
    printf("\n");
    printf("      oiter");
    printf(" iiter");
    printf("        E(p)");
    printf("        E(d)");
    printf("      E(gap)");
    printf("      mu");
    printf("     eps(p)");
    printf("     eps(d)\n");

    double primal_dual_energy_gap = 0.0;

    int oiter_local = 0;

    // TODO: get MU_UPDATE_FREQUENCY from set function
    int mu_update_frequency = 500;

    do {

        double start = omp_get_wtime();

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
        // TODO: get CG_CONVERGENCE from set() function
        double cg_conv_i = 1e-8;
        if (oiter_ == 0)
            cg_conv_i = 0.01;
        else
            cg_conv_i = (primal_error_ > dual_error_) ? 0.01 * dual_error_ : 0.01 * primal_error_;
        if (cg_conv_i < cg_convergence)
            cg_conv_i = cg_convergence;
        cg->set_convergence(cg_conv_i);

        // solve CG problem (step 1 in table 1 of PRL 106 083001)
        cg->solve(Au_,y_,cg_rhs_,evaluate_cg_AATu,(void*)this);
        int iiter = cg->total_iterations();

        double end = omp_get_wtime();

        iiter_time_  += end - start;
        iiter_total_ += iiter;

        start = omp_get_wtime();

        // update primal and dual solutions
        Update_xz(x, c, primal_block_dim, evaluate_ATu, data);

        end = omp_get_wtime();

        oiter_time_ += end - start;

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
        double energy_primal = C_DDOT(n_primal_,c,1,x,1);
        double energy_dual   = C_DDOT(n_dual_,b,1,y_,1);

        primal_dual_energy_gap = fabs(energy_primal-energy_dual);

        printf("      %5i %5i %11.6lf %11.6lf %11.6le %7.3lf %10.5le %10.5le\n",
                    oiter_,iiter,energy_primal,energy_dual,primal_dual_energy_gap,mu_,primal_error_,dual_error_);

        oiter_++;
        oiter_local++;

        // don't update mu every iteration
        if ( oiter_ % mu_update_frequency == 0 && oiter_ > 0 ){
            mu_ = mu_ * primal_error_ / dual_error_;
        }

        if ( primal_error_ > r_convergence_ || dual_error_ > r_convergence_  || primal_dual_energy_gap > e_convergence_ ) {
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
    C_DSCAL(n_primal_,mu_,x,1);
    C_DAXPY(n_primal_,1.0,x,1,ATu_,1);

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

        double * mat     = (double*)malloc(primal_block_dim[i]*primal_block_dim[i]*sizeof(double));
        double * eigvec  = (double*)malloc(primal_block_dim[i]*primal_block_dim[i]*sizeof(double));
        double * eigval  = (double*)malloc(primal_block_dim[i]*sizeof(double));

        for (int p = 0; p < primal_block_dim[i]; p++) {
            for (int q = p; q < primal_block_dim[i]; q++) {
                double dum = 0.5 * ( ATu_[myoffset + p * primal_block_dim[i] + q] +
                                     ATu_[myoffset + q * primal_block_dim[i] + p] );
                mat[p*primal_block_dim[i] + q] = mat[q*primal_block_dim[i] + p] = dum;
            }
        }

        Diagonalize(primal_block_dim[i],mat,eigval);
        C_DCOPY(primal_block_dim[i]*primal_block_dim[i],mat,1,eigvec,1);

        // separate U+ and U-, transform back to nondiagonal basis

        double * eigvec2 = (double*)malloc(primal_block_dim[i]*primal_block_dim[i]*sizeof(double));

        // (+) part
        long int mydim = 0;
        for (long int j = 0; j < primal_block_dim[i]; j++) {
            if ( eigval[j] > 0.0 ) {
                for (long int q = 0; q < primal_block_dim[i]; q++) {
                    mat[q*primal_block_dim[i]+mydim]   = eigvec[q*primal_block_dim[i]+j] * eigval[j]/mu_;
                    eigvec2[q*primal_block_dim[i]+mydim] = eigvec[q*primal_block_dim[i]+j];
                }
                mydim++;
            }
        }

        long int block_dim = primal_block_dim[i];

        F_DGEMM(&char_t,&char_n,&block_dim,&block_dim,&mydim,&one,mat,&block_dim,eigvec2,&block_dim,&zero,&x[myoffset],&block_dim);

        // (-) part
        mydim = 0;
        for (long int j = 0; j < block_dim; j++) {
            if ( eigval[j] < 0.0 ) {
                for (long int q = 0; q < block_dim; q++) {
                    mat[q*block_dim+mydim]   = -eigvec[q*block_dim+j] * eigval[j];
                    eigvec2[q*block_dim+mydim] =  eigvec[q*block_dim+j];
                }
                mydim++;
            }
        }
        F_DGEMM(&char_t,&char_n,&block_dim,&block_dim,&mydim,&one,mat,&block_dim,eigvec,&block_dim,&zero,&z_[myoffset],&block_dim);

        free(mat);
        free(eigvec);
        free(eigvec2);
        free(eigval);

    }

}

}

