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

#include<string.h>

#include "sdp_solver.h"

namespace libsdp {

SDPSolver::SDPSolver(long int n_primal, long int n_dual){

    n_primal_     = n_primal;
    n_dual_       = n_dual;
    mu_           = 0.1;
    primal_error_ = 0.0;
    dual_error_   = 0.0;
    oiter_        = 0;
    iiter_total_  = 0;
    oiter_time_   = 0.0;
    iiter_time_   = 0.0;

    y_   = (double*)malloc(n_dual_   * sizeof(double));
    Au_  = (double*)malloc(n_dual_   * sizeof(double));
    z_   = (double*)malloc(n_primal_ * sizeof(double));
    ATu_ = (double*)malloc(n_primal_ * sizeof(double));

    memset((void*)y_,  '\0',n_dual_   * sizeof(double));
    memset((void*)Au_, '\0',n_dual_   * sizeof(double));
    memset((void*)z_,  '\0',n_primal_ * sizeof(double));
    memset((void*)ATu_,'\0',n_primal_ * sizeof(double));

    // TODO handle with set_...
    e_convergence_ = 1.0e-5; 
    r_convergence_ = 1.0e-5; 

    is_converged_ = false;
}

SDPSolver::~SDPSolver(){

    free(y_);
    free(Au_);
    free(z_);
    free(ATu_);

}

}

