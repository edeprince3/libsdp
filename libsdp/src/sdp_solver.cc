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

void SDPSolver::write_xyz(double * x) {

    // user may not want to write solution to disk
    if ( options_.outfile == "" ) {
        return;
    }

    FILE * fp = fopen(options_.outfile.c_str(), "w");

    // x
    fprintf(fp, "%li\n", n_primal_);
    for (long int i = 0; i < n_primal_; i++) {
        fprintf(fp, "%20.12le\n", x[i]);
    }

    // y
    fprintf(fp, "%li\n", n_dual_);
    for (long int i = 0; i < n_dual_; i++) {
        fprintf(fp, "%20.12le\n", y_[i]);
    }
    
    // z
    fprintf(fp, "%li\n", n_primal_);
    for (long int i = 0; i < n_primal_; i++) {
        fprintf(fp, "%20.12le\n", z_[i]);
    }

    // mu
    fprintf(fp, "%20.12le\n", mu_);
    
    fclose(fp);
}

void SDPSolver::read_xyz(double * x) {

    FILE * fp = fopen(options_.outfile.c_str(), "r");
    if (fp == NULL) { 
        printf("\n");
        printf("    error: restart file does not exist: %s\n", options_.outfile.c_str());
        printf("\n");
        exit(1);
    }

    long int dim;

    // x
    fscanf(fp, "%li", &dim);
    if ( dim != n_primal_ ) {
        printf("\n");
        printf("    error: dimension mismatch when reading solution (x)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le", &x[i]);
    }

    // y
    fscanf(fp, "%li", &dim);
    if ( dim != n_dual_ ) {
        printf("\n");
        printf("    error: dimension mismatch when reading solution (y)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le", &y_[i]);
    }

    // x
    fscanf(fp, "%li", &dim);
    if ( dim != n_primal_ ) {
        printf("\n");
        printf("    error: dimension mismatch when reading solution (z)\n");
        printf("\n");
        exit(1);
    }
    for (long int i = 0; i < dim; i++) {
        fscanf(fp, "%le", &z_[i]);
    }

    fscanf(fp, "%le", &mu_);

    fclose(fp);
}

}

