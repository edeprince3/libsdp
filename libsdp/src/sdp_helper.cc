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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

#include "sdp_helper.h"
#include <rrsdp_solver.h>
#include <bpsdp_solver.h>

#include "blas_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

std::vector<int> empty_list = {};

namespace libsdp {

void export_SDPHelper(py::module& m) {

    // export SDP options
    py::class_<SDPOptions> options(m, "sdp_options");

    options.def(py::init< >())
        .def_readwrite("lbfgs_maxiter",&SDPOptions::lbfgs_maxiter)
        .def_readwrite("cg_maxiter",&SDPOptions::cg_maxiter)
        .def_readwrite("mu_update_frequency",&SDPOptions::mu_update_frequency)
        .def_readwrite("cg_convergence",&SDPOptions::cg_convergence)
        .def_readwrite("dynamic_cg_convergence",&SDPOptions::dynamic_cg_convergence)
        .def_readwrite("sdp_objective_convergence",&SDPOptions::sdp_objective_convergence)
        .def_readwrite("sdp_error_convergence",&SDPOptions::sdp_error_convergence)
        .def_readwrite("sdp_objective_convergence",&SDPOptions::sdp_objective_convergence)
        .def_readwrite("penalty_parameter_scaling",&SDPOptions::penalty_parameter_scaling)
        .def_readwrite("penalty_parameter",&SDPOptions::penalty_parameter)
        .def_readwrite("print_level",&SDPOptions::print_level)
        .def_readwrite("guess_type",&SDPOptions::guess_type)
        .def_readwrite("outfile",&SDPOptions::outfile)
        .def_readwrite("sdp_algorithm",&SDPOptions::algorithm)
        .def_readwrite("procedure",&SDPOptions::procedure);

    // export SDPMatrix type
    py::class_<SDPMatrix>(m, "sdp_matrix")
        .def(py::init<>())
        .def_readwrite("block_number", &SDPMatrix::block_number)
        .def_readwrite("row", &SDPMatrix::row)
        .def_readwrite("column", &SDPMatrix::column)
        .def_readwrite("value", &SDPMatrix::value)
        .def_readwrite("id", &SDPMatrix::id);

    // export SDP solver
    py::class_<SDPHelper, std::shared_ptr<SDPHelper> >(m, "sdp_solver")
        .def(py::init<SDPOptions, std::vector<SDPMatrix> &, std::vector<int> & >())
        .def("solve", 
             [](SDPHelper& self, 
                 const std::vector<double> & b, 
                 const int & maxiter,
                 const std::vector<int> & primal_block_rank) {
                 return self.solve(b, maxiter, primal_block_rank);
             },
             py::arg("b"), 
             py::arg("maxiter"), 
             py::arg("primal_block_rank") = empty_list )
        .def("get_ATu", &SDPHelper::get_ATu)
        .def("get_Au", &SDPHelper::get_Au)
        .def("get_y", &SDPHelper::get_y)
        .def("get_z", &SDPHelper::get_z)
        .def("get_c", &SDPHelper::get_c)
        .def("get_mu", &SDPHelper::get_mu);
}

PYBIND11_MODULE(_libsdp, m) {
    m.doc() = "Python API of libsdp";
    export_SDPHelper(m);
}

/// BPSDP monitor callback function
static void bpsdp_monitor(int print_level, int oiter, int iiter, double energy_primal, double energy_dual, double mu, double primal_error, double dual_error, void * data) {

    if ( print_level > 0 ) {
        if ( oiter % print_level == 0 ) {
            printf("      %5i %5i %11.6lf %11.6lf %11.6le %7.3lf %10.5le %10.5le\n",
                oiter,iiter,energy_primal,energy_dual,fabs(energy_primal-energy_dual),mu,primal_error,dual_error);
            fflush(stdout);
        }
    }
}

/// RRSDP monitor callback function
static void rrsdp_monitor(int print_level, int oiter, int iiter, double lagrangian, double objective, double mu, double error, double zero, void * data) {

    if ( print_level > 0 ) {
        if ( oiter % print_level == 0 ) {
            printf("    %12i %12i %12.6lf %12.6lf %12.2le %12.3le\n",
                oiter,iiter,lagrangian,objective,mu,error);
            fflush(stdout);
        }
    }
}

/// SDPHelper constructor
SDPHelper::SDPHelper(SDPOptions options,
                     std::vector<SDPMatrix> Fi,
                     std::vector<int> primal_block_dim) {

    options_      = options;

    std::transform(options_.procedure.begin(), options_.procedure.end(), options_.procedure.begin(),
        [](unsigned char c){ return std::tolower(c); });

    std::transform(options_.algorithm.begin(), options_.algorithm.end(), options_.algorithm.begin(),
        [](unsigned char c){ return std::tolower(c); });

    std::transform(options_.guess_type.begin(), options_.guess_type.end(), options_.guess_type.begin(),
        [](unsigned char c){ return std::tolower(c); });

    // copy some quantities to class members for objective 
    // function / Au / ATu evaluation

    // number of primal variables
    n_primal_ = 0;
    for (size_t i = 0; i < primal_block_dim.size(); i++) {
        n_primal_ += primal_block_dim[i] * primal_block_dim[i];
    }

    // number of dual variables (F0 is part of Fi...)
    n_dual_ = Fi.size()-1;

    // c vector (-F0 in SDPA format)

    c_.resize(n_primal_);

    // note that, compared to SDPLIB problems, our definition of the problem 
    // has c = -F0. (we're minimizing; they maximize). the result will be that
    // our final objective will have the opposite sign compared to tabulated
    // SDPLIB values

    int sign = 1;
    if ( options_.procedure == "minimize" ) {
    }else if ( options_.procedure == "maximize" ) {
        sign = -1;
    }else {
        printf("\n");
        printf("    error: invalid procedure: %s\n", options_.procedure.c_str());
        printf("\n");
        exit(1);
    }

    for (size_t i = 0; i < Fi[0].block_number.size(); i++) {
        int my_block  = Fi[0].block_number[i] - 1;
        int my_row    = Fi[0].row[i] - 1;
        int my_column = Fi[0].column[i] - 1;

        // calculate offset
        size_t off = 0;
        for (size_t j = 0; j < my_block; j++) {
            off += primal_block_dim[j] * primal_block_dim[j];
        }

        // populate relevant entry in c. 
        c_[off + my_row * primal_block_dim[my_block] + my_column] += sign * Fi[0].value[i];
    }

    // constraint matrices
    for (size_t i = 1; i < Fi.size(); i++) {

        Fi_.push_back(Fi[i]);

        // add composite indices
        for (size_t j = 0; j < Fi[i].block_number.size(); j++) {

            // don't forget that the input Fi used unit-offset labels
            int my_block  = Fi[i].block_number[j] - 1;
            int my_row    = Fi[i].row[j] - 1;
            int my_column = Fi[i].column[j] - 1;

            Fi_[i-1].block_number[j] = my_block;
            Fi_[i-1].row[j]          = my_row;
            Fi_[i-1].column[j]       = my_column;

            // calculate offset
            size_t off = 0;
            for (size_t k = 0; k < my_block; k++) {
                off += primal_block_dim[k] * primal_block_dim[k];
            }

            // composite index
            size_t id = off + my_row * primal_block_dim[my_block] + my_column;

            // add to matrix object
            //Fi_[i-1].id.append(id);
            Fi_[i-1].id.push_back(id);
        }
    }

    // FTi matrices:
    FTi_.resize(n_primal_);
    for (size_t i = 0; i < Fi_.size(); i++) {
        for (size_t j = 0; j < Fi_[i].block_number.size(); j++) {
            //FTi_[Fi_[i].id[j]].id.append(i);
            //FTi_[Fi_[i].id[j]].value.append(Fi_[i].value[j]);
            FTi_[Fi_[i].id[j]].id.push_back(i);
            FTi_[Fi_[i].id[j]].value.push_back(Fi_[i].value[j]);
        }
    }

    // primal block dimensions
    for (size_t i = 0; i < primal_block_dim.size(); i++) {
        primal_block_dim_.push_back(primal_block_dim[i]);
    }

    // initialize sdp solver

    if ( options_.algorithm != "bpsdp" && options_.algorithm != "rrsdp" ) {
        printf("\n");
        printf("    error: undefined SDP algorithm: %s\n", options_.algorithm.c_str());
        printf("\n");
        exit(1);
    }

    if ( options_.algorithm == "bpsdp" ) {

        sdp_ = (std::shared_ptr<SDPSolver>)(new BPSDPSolver(n_primal_,n_dual_,options_));
        sdp_monitor_ = bpsdp_monitor;

    }else if ( "rrsdp" ) {

        sdp_ = (std::shared_ptr<SDPSolver>)(new RRSDPSolver(n_primal_,n_dual_,options_));
        sdp_monitor_ = rrsdp_monitor;

    }

    // primal solution vector (random guess on [-0.001:0.001])
    x_.clear();

    if ( options_.guess_type == "random" ) {
        srand(0);
        for (size_t i = 0; i < n_primal_; i++) {
            x_.push_back(2.0 * ( (double)rand()/RAND_MAX - 0.5 ) * 0.001);
        }
    }else if ( options_.guess_type == "zero" ) {
        if ( options_.algorithm == "rrsdp" ) {
            printf("\n");
            printf("    error: guess_type = 'zero' not valid for SDP algorithm: rrsdp\n");
            printf("\n");
            exit(1);
        }
        for (size_t i = 0; i < n_primal_; i++) {
            x_.push_back(0.0);
        }
    }else if ( options_.guess_type == "read"  ) {
        for (size_t i = 0; i < n_primal_; i++) {
            x_.push_back(0.0);
        }
        sdp_->read_xyz(x_.data());
    }else {
        printf("\n");
        printf("    error: undefined guess type: %s\n", options_.guess_type.c_str());
        printf("\n");
        exit(1);
    }
}

/// SDPHelper destructor
SDPHelper::~SDPHelper() {
}

/// SDP callback function: Au
static void Au_callback(double * Au, double * u, void * data) {

    // reinterpret void * as an instance of SDPHelper
    SDPHelper * sdp = reinterpret_cast<SDPHelper*>(data);
    sdp->evaluate_Au(Au,u);

}
void SDPHelper::evaluate_Au(double * Au, double * u) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Fi_.size(); i++) {
        double dum = 0.0;
        for (size_t j = 0; j < Fi_[i].block_number.size(); j++) {
            dum += Fi_[i].value[j] * u[Fi_[i].id[j]];
        }
        Au[i] = dum;
    }
}

/// SDP callback function: ATu
static void ATu_callback(double * ATu, double * u, void * data) {

    // reinterpret void * as an instance of SDPHelper
    SDPHelper * sdp = reinterpret_cast<SDPHelper*>(data);
    sdp->evaluate_ATu(ATu,u);

}
void SDPHelper::evaluate_ATu(double * ATu, double * u) {
    //memset((void*)ATu,'\0',n_primal_*sizeof(double));
    //for (size_t i = 0; i < Fi_.size(); i++) {
    //    for (size_t j = 0; j < Fi_[i].block_number.size(); j++) {
    //        ATu[Fi_[i].id[j]] += Fi_[i].value[j] * u[i];
    //    }
    //}
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < FTi_.size(); i++) {
        double dum = 0.0;
        for (size_t j = 0; j < FTi_[i].id.size(); j++) {
            dum += FTi_[i].value[j] * u[FTi_[i].id[j]];
        }
        ATu[i] = dum;
    }
}

/// solve the sdp problem and return the primal solution
std::vector<double> SDPHelper::solve(std::vector<double> b, 
                                     int maxiter,
                                     std::vector<int> primal_block_rank) {

    // print header
    if ( options_.algorithm == "bpsdp" ) {
    
        if ( !primal_block_rank.empty() ) {
            printf("\n");
            printf("    ==> WARNING <== \n");
            printf("\n");
            printf("        SDP algorithm BPSDP uses full-rank RDMs\n");
            printf("\n");
        }

        if ( options_.print_level > 0 ) {
            printf("\n");
            printf("    ==> BPSDP: Boundary-point SDP <==\n");
            printf("\n");
            printf("      oiter");
            printf(" iiter");
            printf("         c.x");
            printf("         b.y");
            printf("    |c.x-b.y|");
            printf("      mu");
            printf("    ||Ax-b||");
            printf(" ||ATy-c+z||\n");
        }

        // solve sdp
        sdp_->solve(x_.data(),
                    b.data(),
                    c_.data(),
                    primal_block_dim_, 
                    maxiter, 
                    Au_callback, 
                    ATu_callback, 
                    sdp_monitor_, 
                    options_.print_level, 
                    (void*)this);
            
    }else if ( options_.algorithm == "rrsdp" ) {
    
        if ( options_.print_level > 0 ) {
            printf("\n");
            printf("    ==> RRSDP: Matrix-factorization-based first-order SDP <==\n");
            printf("\n");
            printf("           oiter");
            printf("        iiter");
            printf("            L");
            printf("          c.x");
            printf("           mu");
            printf("     ||Ax-b||\n");
        }
        
        // primal block ranks
        if ( primal_block_rank.empty() ) {
            for (size_t i = 0; i < primal_block_dim_.size(); i++) {
                primal_block_rank_.push_back(primal_block_dim_[i]);
            }
        }else {
            for (size_t i = 0; i < primal_block_rank.size(); i++) {
                primal_block_rank_.push_back(primal_block_rank[i]);
            }
        }

        // solve (possibly low-rank) sdp
        sdp_->solve_low_rank(x_.data(),
                             b.data(),
                             c_.data(),
                             primal_block_dim_, 
                             primal_block_rank_, 
                             maxiter, 
                             Au_callback, 
                             ATu_callback, 
                             sdp_monitor_, 
                             options_.print_level, 
                             (void*)this);
    }


    if ( options_.print_level > 0 ) {
        printf("\n");
        fflush(stdout);
    }

    return x_;
}

/// return the BPSDP z dual variable
std::vector<double> SDPHelper::get_z() {
    if ( options_.algorithm != "bpsdp" ) {
        printf("\n");
        printf("    error: z dual variable only defined for SDP algorithm BPSDP\n");
        printf("\n");
    }
    double * tmp_z = sdp_->get_z();
    std::vector<double> z(tmp_z, tmp_z + n_primal_);
    return z;
}

/// return the BPSDP y dual variable or RRSDP lagrange multipliers
std::vector<double> SDPHelper::get_y() {
    double * tmp_y = sdp_->get_y();
    std::vector<double> y(tmp_y, tmp_y + n_dual_);
    return y;
}

/// return the RRSDP penalty parameter, mu
double SDPHelper::get_mu() {
    return sdp_->get_mu();
}

/// return the action of A^T on a vector
std::vector<double> SDPHelper::get_ATu(std::vector<double> u) {

    double * tmp_ATu = (double*)malloc(n_primal_*sizeof(double));

    evaluate_ATu(tmp_ATu, u.data() );

    std::vector<double> ATu(tmp_ATu, tmp_ATu + n_primal_);

    free(tmp_ATu);

    return ATu;
}

/// return the action of A on a vector
std::vector<double> SDPHelper::get_Au(std::vector<double> u) {

    double * tmp_Au = (double*)malloc(n_dual_*sizeof(double));

    evaluate_Au(tmp_Au, u.data() );

    std::vector<double> Au(tmp_Au, tmp_Au + n_dual_);

    free(tmp_Au);

    return Au;
}

SDPOptions options() {
    SDPOptions opt;
    return opt;
}

}
