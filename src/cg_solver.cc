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

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "cg_solver.h"

namespace libsdp{

CGSolver::CGSolver(long int n) {
    n_              = n;
    iter_           = 0;
    cg_max_iter_    = 10000;
    cg_convergence_ = 1e-9;
    p_ = (double*)malloc(n_*sizeof(double));
    r_ = (double*)malloc(n_*sizeof(double));
}

CGSolver::~CGSolver(){
    free(p_);
    free(r_);
}

void CGSolver::set_max_iter(int iter) {
    cg_max_iter_ = iter;
}

void CGSolver::set_convergence(double conv) {
    cg_convergence_ = conv;
}

void CGSolver::solve(double * Ap, 
                     double *  x, 
                     double *  b, 
                     CGCallbackFunction function, void * data) {

    double alpha = 0.0;
    double beta  = 0.0;

    // call some function to evaluate A.x.  Result in Ap
    function(Ap,x,data);

    // r = b - Ap
    C_DCOPY(n_,b,1,r_,1);
    C_DAXPY(n_,-1.0,Ap,1,r_,1);

    C_DCOPY(n_,r_,1,p_,1);

    iter_ = 0;
    do {

        // call some function to evaluate A.p.  Result in Ap
        function(Ap,p_,data);

        double rr  = C_DDOT(n_,r_,1,r_,1);
        double pap = C_DDOT(n_,p_,1,Ap,1);
        double alpha = rr / pap;
        C_DAXPY(n_,alpha,p_,1,x,1);
        C_DAXPY(n_,-alpha,Ap,1,r_,1);

        // if r is sufficiently small, then exit loop
        double rrnew = C_DDOT(n_,r_,1,r_,1);
        double nrm = sqrt(rrnew);
        double beta = rrnew/rr;
        if ( nrm < cg_convergence_ ) break;

        C_DSCAL(n_,beta,p_,1);
        C_DAXPY(n_,1.0,r,1,p_,1);

        iter_++;

    }while(iter_ < cg_max_iter_ );
}

int CGSolver::total_iterations() {
    return iter_;
}

}// end of namespace
