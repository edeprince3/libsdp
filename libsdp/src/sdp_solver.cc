/* 
 *  @BEGIN LICENSE
 * 
 *  libsdp: a library of semidefinite programming solvers
 * 
 *  Copyright (c) 2021-2024 by A. E. DePrince III
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

#include<string.h>

#include <sdp_solver.h>

#include "blas_helper.h"

namespace libsdp {

SDPSolver::SDPSolver(long int n_primal, long int n_dual, SDPOptions options){

    options_      = options;
    n_primal_     = n_primal;
    n_dual_       = n_dual;
    mu_           = 0.1;
    primal_error_ = 0.0;
    dual_error_   = 0.0;
    oiter_        = 0;
    iiter_total_  = 0;

    y_   = (double*)malloc(n_dual_   * sizeof(double));
    Au_  = (double*)malloc(n_dual_   * sizeof(double));
    z_   = (double*)malloc(n_primal_ * sizeof(double));
    ATu_ = (double*)malloc(n_primal_ * sizeof(double));

    memset((void*)y_,  '\0',n_dual_   * sizeof(double));
    memset((void*)Au_, '\0',n_dual_   * sizeof(double));
    memset((void*)z_,  '\0',n_primal_ * sizeof(double));
    memset((void*)ATu_,'\0',n_primal_ * sizeof(double));

    is_converged_ = false;
}

SDPSolver::~SDPSolver(){

    free(y_);
    free(Au_);
    free(z_);
    free(ATu_);

}

void SDPSolver::set_y(double * y) { 
    C_DCOPY(n_dual_,y,1,y_,1); 
}

void SDPSolver::set_z(double * z) {
   C_DCOPY(n_primal_,z,1,z_,1); 
}

}

