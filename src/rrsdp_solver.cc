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

#include <rrsdp_solver.h>

#include "blas_helper.h"
#include "lbfgs_helper.h"

namespace libsdp {

// liblbfgs routines:
static lbfgsfloatval_t lbfgs_evaluate(void * instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step) {

    RRSDPSolver * sdp = reinterpret_cast<RRSDPSolver*>(instance);
    double f = sdp->evaluate_gradient_x(x,g);

    return f;
}

static int monitor_lbfgs_progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    RRSDPSolver * sdp = reinterpret_cast<RRSDPSolver*>(instance);
    sdp->set_iiter(k);
    return 0;
}

RRSDPSolver::RRSDPSolver(long int n_primal, long int n_dual, SDPOptions options)
    : SDPSolver(n_primal,n_dual,options) {

    iiter_ = 0;

    // lbfgs container auxiliary variables that define x
    lbfgs_vars_x_ = lbfgs_malloc(n_primal_);

    // seed auxiliary variables
    srand(0);
    for (int i = 0; i < n_primal_; i++) {
        lbfgs_vars_x_[i] = 2.0 * ( (double)rand()/RAND_MAX - 0.5 ) / 1000.0;
    }

    mu_ = 0.1;
    mu_reset_ = true;
    mu_scale_factor_ = options.penalty_parameter_scaling;
}

RRSDPSolver::~RRSDPSolver(){

    free(lbfgs_vars_x_);

}

void RRSDPSolver::solve(double * x,   
                        double * b, 
                        double * c,
                        std::vector<int> primal_block_dim,
                        int maxiter,
                        SDPCallbackFunction evaluate_Au, 
                        SDPCallbackFunction evaluate_ATu, 
                        SDPProgressMonitorFunction progress_monitor, 
                        void * data){

    // class pointer to input data
    data_ = data;

    // class pointers to callback functions
    evaluate_Au_  = evaluate_Au;
    evaluate_ATu_ = evaluate_ATu;

    // copy block sizes
    primal_block_dim_.clear();
    for (int block = 0; block < primal_block_dim.size(); block++) {
        primal_block_dim_.push_back(primal_block_dim[block]);
    }
    // if block ranks are not set, assign rank = dim
    if ( primal_block_rank_.size() != primal_block_dim_.size() ) {
        primal_block_rank_.clear();
        for (int block = 0; block < primal_block_dim.size(); block++) {
            primal_block_rank_.push_back(primal_block_dim[block]);
        }
    }

    // class pointers to input c,x,b
    c_ = c;
    x_ = x;
    b_ = b;

    build_x(lbfgs_vars_x_);

    // initial objective function value   
    double objective =  C_DDOT(n_primal_,x_,1,c_,1);

    // this function can be called many times. don't forget to reset penalty parameter
    if ( mu_reset_ ) {
        mu_ = 0.1;
    }

    int oiter_local = 0;

    double max_err = -999;

    double * tmp = (double*)malloc(n_primal_*sizeof(double));
    memset((void*)tmp,'\0',n_primal_*sizeof(double));

    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = options_.maxiter;
    //param.epsilon        = 1e-8;

    do {

        // minimize lagrangian

        // initial objective function value default parameters
        lbfgsfloatval_t lagrangian = evaluate_gradient_x(lbfgs_vars_x_,tmp);

        if (oiter_ == 0) {
            param.epsilon = 0.01;
        }else {
            param.epsilon = 0.01 * primal_error_;
        }
        if ( param.epsilon < options_.sdp_error_convergence ) {
            param.epsilon = options_.sdp_error_convergence;
        }
        int status = lbfgs(n_primal_,lbfgs_vars_x_,&lagrangian,lbfgs_evaluate,monitor_lbfgs_progress,(void*)this,&param);
        //lbfgs_error_check(status);

        // update lagrange multipliers and penalty parameter

        // build x = r.rT
        build_x(lbfgs_vars_x_);

        // evaluate x^T.c
        double objective_primal = C_DDOT(n_primal_,x_,1,c_,1);

        // evaluate (Ax-b)
        evaluate_Au_(Au_,x_,data_);
        C_DAXPY(n_dual_,-1.0,b_,1,Au_,1);
        primal_error_ = C_DNRM2(n_dual_,Au_,1);

        double new_max_err = 0.0;
        int imax = -1;
        //for (int i = 0; i < n_dual_; i++) {
        //    if ( fabs(Au_[i]) > new_max_err ) {
        //        new_max_err = fabs(Au_[i]);
        //        imax = i;
        //     }
        //}
        //if ( new_max_err < 0.25 * max_err ){
            for (int i = 0; i < n_dual_; i++) {
                y_[i] -= Au_[i] / mu_;
            }
        //}else{
            mu_ *= mu_scale_factor_;
        //}
        max_err = new_max_err;

        progress_monitor(oiter_,iiter_,lagrangian,objective_primal,mu_,primal_error_,0.0, data);

        iiter_total_ += iiter_;

        oiter_++;
        oiter_local++;

        if ( primal_error_ > options_.sdp_error_convergence  || fabs(objective - objective_primal) > options_.sdp_objective_convergence ) {
            is_converged_ = false;
        }else {
            is_converged_ = true;
        }

        // update objective function value
        objective = objective_primal;

        if ( oiter_local == maxiter ) break;

    }while( !is_converged_ );

}

// build x = r.rT
void RRSDPSolver::build_x(double * r){

    char char_n = 'n';
    char char_t = 't';
    double one = 1.0;
    double zero = 0.0;

    long int off_nn = 0;
    long int off_nm = 0;
    for (int block = 0; block < primal_block_dim_.size(); block++) {
        long int n = primal_block_dim_[block];
        long int m = primal_block_rank_[block];
        if ( n == 0 ) continue;
        F_DGEMM(&char_n, &char_t, &n, &n, &m, &one, &r[off_nm], &n, &r[off_nm], &n, &zero, &x_[off_nn], &n);
        off_nm += n*m;
        off_nn += n*n;
    }

}

double RRSDPSolver::evaluate_gradient_x(const lbfgsfloatval_t * r, lbfgsfloatval_t * g) {

    // L = x^Tc - y^T (Ax - b) + 1/mu || Ax - b ||

    double * r_p   = (double*)r;

    // build x = r.rT
    build_x(r_p);

    // evaluate primal objective function value
    double objective = C_DDOT(n_primal_,x_,1,c_,1);

    // evaluate (Ax-b)
    evaluate_Au_(Au_,x_,data_);
    C_DAXPY(n_dual_,-1.0,b_,1,Au_,1);

    // evaluate sqrt(||Ax-b||)
    double nrm = C_DNRM2(n_dual_,Au_,1);

    // evaluate lagrangian
    double lagrangian = objective - C_DDOT(n_dual_,y_,1,Au_,1) + nrm*nrm/(2.0 * mu_);

    // dL/dR = 2( A^T [ 2/mu(Ax-b) - y] + c) . r

    // evaluate A^T (1/mu[Ax-b] - y)
    C_DSCAL(n_dual_,1.0/mu_,Au_,1);
    C_DAXPY(n_dual_,-1.0,y_,1,Au_,1);
    evaluate_ATu_(ATu_,Au_,data_);

    // add integrals for derivative of objective function value
    C_DAXPY(n_primal_,1.0,c_,1,ATu_,1);

    int off_nn = 0;
    for (int block = 0; block < primal_block_dim_.size(); block++) {
        int n = primal_block_dim_[block];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double dum_ij = ATu_[i*n+j + off_nn];
                double dum_ji = ATu_[j*n+i + off_nn];
                ATu_[i*n+j + off_nn] = dum_ij + dum_ji;
                ATu_[j*n+i + off_nn] = dum_ij + dum_ji;
            }
        }
        off_nn += n*n;
    }

    // evaluate gradient of lagrangian

    char char_n = 'n';
    double one = 1.0;
    double zero = 0.0;

    off_nn = 0;
    long int off_nm = 0;
    for (int block = 0; block < primal_block_dim_.size(); block++) {
        long int n = primal_block_dim_[block];
        long int m = primal_block_rank_[block];
        if ( n == 0 ) continue;
        F_DGEMM(&char_n, &char_n, &n, &m, &n, &one, &ATu_[off_nn], &n, &r_p[off_nm], &n, &zero, &g[off_nm], &n);
        off_nn += n*n;
        off_nm += n*m;
    }

    return lagrangian;
}


}

