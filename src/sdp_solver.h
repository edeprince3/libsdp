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

#ifndef SDP_SOLVER_H
#define SDP_SOLVER_H

#include<stdio.h>
#include<stdlib.h>

#include<vector>

namespace libsdp {

struct SDPOptions {
    int cg_maxiter                   = 10000;
    int mu_update_frequency          = 500;
    double cg_convergence            = 1e-8;
    double sdp_objective_convergence = 1e-4;
    double sdp_error_convergence     = 1e-4;
};

typedef void (*SDPCallbackFunction)(double *,double *,void *);

class SDPSolver{

  public:

    /// SDPSolver constructor
    SDPSolver(long int n_primal, long int n_dual, SDPOptions options);

    /// SDPSolver destructor
    ~SDPSolver();

    /// solve the sdp problem
    virtual void solve(double * x,
                       double * b,
                       double * c,
                       std::vector<int> primal_block_dim,
                       int maxiter,
                       SDPCallbackFunction evaluate_Au,
                       SDPCallbackFunction evaluate_ATu,
                       void * data){
        printf("\n");
        printf("    solve() has not been implemented for this sdp solver\n");
        printf("\n");
        exit(1);
    }

    int iiter_total() { return iiter_total_; }
    int oiter_total() { return oiter_; }
    double iiter_time() { return iiter_time_; }
    double oiter_time() { return oiter_time_; }

    void set_mu(double mu) { mu_ = mu; }
    void set_y(double * y);
    void set_z(double * z);

    double get_mu() { return mu_; }
    double * get_y() { return y_; }
    double * get_z() { return z_; }

    bool is_converged(){ return is_converged_; }

    virtual void set_mu_scale_factor(double mu_scale_factor) { 
        printf("\n");
        printf("    set_mu_scale_factor() has not been implemented for this sdp solver\n");
        printf("\n");
        exit(1);
    }

    virtual void set_mu_reset(bool mu_reset) { 
        printf("\n");
        printf("    set_mu_reset() has not been implemented for this sdp solver\n");
        printf("\n");
        exit(1);
    }

  protected:

    /// options for the SDP
    SDPOptions options_;

    /// pointer to input data
    void * data_;

    /// copy of Au callback function
    SDPCallbackFunction evaluate_Au_;

    /// copy of ATu callback function
    SDPCallbackFunction evaluate_ATu_;

    /// the error in the primal constraints
    double primal_error_;

    /// the error in the dual constraints
    double dual_error_;

    /// is the solver converged?
    bool is_converged_;

    /// the number of outer iterations
    int oiter_;

    /// the total number of inner iterations
    int iiter_total_;

    /// the total time taken by the inner iterations
    double iiter_time_;

    /// the total time taken by the outer iterations
    double oiter_time_;

    /// the penalty parameter
    double mu_;

    /// the dual solution vector
    double * y_;

    /// the second dual solution vector
    double * z_;

    /// temporary container the size of Au
    double * Au_;

    /// temporary container the size of ATu
    double * ATu_;

    /// the dimension of the primal vector
    long int n_primal_;

    /// the dimension of the dual vector
    long int n_dual_;

};

}

#endif
