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

#ifndef SDP_SOLVER_H
#define SDP_SOLVER_H

#include<stdio.h>
#include<stdlib.h>

#include<vector>
#include<string>
#include<functional>

namespace libsdp {

struct SDPOptions {
    SDPOptions(){};
    int maxiter                      = 50000;
    int cg_maxiter                   = 10000;
    int mu_update_frequency          = 500;
    double penalty_parameter_scaling = 0.1;
    double penalty_parameter         = 0.1;
    double cg_convergence            = 1e-8;
    bool dynamic_cg_convergence      = true;
    double sdp_objective_convergence = 1e-4;
    double sdp_error_convergence     = 1e-4;
    int print_level                  = 1;
    std::string guess_type           = "random";
    std::string algorithm            = "bpsdp";
    std::string procedure            = "minimize";
};

typedef std::function<void(double*,double*,void*)> SDPCallbackFunction;
typedef std::function<void(int,int,int,double,double,double,double,double,void*)> SDPProgressMonitorFunction;

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
                       SDPProgressMonitorFunction progress_monitor,
                       int print_level,
                       void * data){
        printf("\n");
        printf("    solve() has not been implemented for this sdp solver\n");
        printf("\n");
        exit(1);
    }

    /// solve the sdp problem (low rank)
    virtual void solve_low_rank(double * x,
                                double * b,
                                double * c,
                                std::vector<int> primal_block_dim,
                                std::vector<int> primal_block_rank,
                                int maxiter,
                                SDPCallbackFunction evaluate_Au,
                                SDPCallbackFunction evaluate_ATu,
                                SDPProgressMonitorFunction progress_monitor,
                                int print_level,
                                void * data){
        printf("\n");
        printf("    solve_low_rank() has not been implemented for this sdp solver\n");
        printf("\n");
        exit(1);
    }

    int iiter_total() { return iiter_total_; }
    int oiter_total() { return oiter_; }

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

    /// read solution from disk
    void read_xyz(double * x);

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

    /// solution file
    std::string outfile_ = "sdp.out";

    /// write solution to disk
    void write_xyz(double * x);
};

}

#endif
