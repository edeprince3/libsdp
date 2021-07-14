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

#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "sdp_helper.h"
#include <rrsdp_solver.h>
#include <bpsdp_solver.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace libsdp {

void export_SDPHelper(py::module& m) {

    // export SDP options

    py::class_<SDPOptions> options(m, "sdp_options");

    options.def(py::init< >())
        .def_readwrite("maxiter",&SDPOptions::maxiter)
        .def_readwrite("cg_maxiter",&SDPOptions::cg_maxiter)
        .def_readwrite("mu_update_frequency",&SDPOptions::mu_update_frequency)
        .def_readwrite("cg_convergence",&SDPOptions::cg_convergence)
        .def_readwrite("sdp_objective_convergence",&SDPOptions::sdp_objective_convergence)
        .def_readwrite("sdp_error_convergence",&SDPOptions::sdp_error_convergence)
        .def_readwrite("sdp_algorithm",&SDPOptions::algorithm);

    py::enum_<SDPOptions::SDPAlgorithm>(options, "SDPAlgorithm")
        .value("RRSDP", SDPOptions::SDPAlgorithm::RRSDP)
        .value("BPSDP", SDPOptions::SDPAlgorithm::BPSDP)
        .export_values();

    // export SDPMatrix type
    py::class_<SDPMatrix> matrix(m, "sdp_matrix");

    matrix.def(py::init< >())
        .def_readwrite("block_number",&SDPMatrix::block_number)
        .def_readwrite("row",&SDPMatrix::row)
        .def_readwrite("column",&SDPMatrix::column)
        .def_readwrite("value",&SDPMatrix::value);

    // export SDP solver

    py::class_<SDPHelper, std::shared_ptr<SDPHelper> >(m, "sdp_solver")
        .def(py::init<long int,long int, SDPOptions>())
        .def("solve", &SDPHelper::solve,
            "b"_a,
            "F0"_a,
            "Fi"_a,
            "primal_block_dim"_a,
            "maxiter"_a);
}

PYBIND11_MODULE(libsdp, m) {
    m.doc() = "Python API of libsdp";
    export_SDPHelper(m);
}


/// SDPHelper constructor
SDPHelper::SDPHelper(long int n_primal, long int n_dual, SDPOptions options) {

    options_      = options;
    n_primal_     = n_primal;
    n_dual_       = n_dual;

}

/// SDPHelper destructor
SDPHelper::~SDPHelper() {
}

/// BPSDP monitor callback function
static void bpsdp_monitor(int oiter, int iiter, double energy_primal, double energy_dual, double mu, double primal_error, double dual_error, void * data) {

    printf("      %5i %5i %11.6lf %11.6lf %11.6le %7.3lf %10.5le %10.5le\n",
        oiter,iiter,energy_primal,energy_dual,fabs(energy_primal-energy_dual),mu,primal_error,dual_error);
    fflush(stdout);

}

/// RRSDP monitor callback function
static void rrsdp_monitor(int oiter, int iiter, double lagrangian, double objective, double mu, double error, double zero, void * data) {

    printf("    %12i %12i %12.6lf %12.6lf %12.2le %12.3le\n",
        oiter,iiter,lagrangian,objective,mu,error);
    fflush(stdout);

}


/// SDP callback function: Au
static void Au_callback(double * Au, double * u, void * data) {

    // reinterpret void * as an instance of SDPHelper
    SDPHelper * sdp = reinterpret_cast<SDPHelper*>(data);
    sdp->evaluate_Au(Au,u);

}
void SDPHelper::evaluate_Au(double * Au, double * u) {
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
    memset((void*)ATu,'\0',n_primal_*sizeof(double));
    for (size_t i = 0; i < Fi_.size(); i++) {
        for (size_t j = 0; j < Fi_[i].block_number.size(); j++) {
            ATu[Fi_[i].id[j]] += Fi_[i].value[j] * u[i];
        }
    }
}

/// solve the sdp problem
void SDPHelper::solve(std::vector<double> b,
                      SDPMatrix F0,
                      std::vector<SDPMatrix> Fi,
                      std::vector<int> primal_block_dim,
                      int maxiter) {

    // copy some quantities to class members for objective 
    // function / Au / ATu evaluation

    // c vector (-F0 in SDPA format)

    std::vector<double> c (n_primal_, 0.0);

    for (size_t i = 0; i < F0.block_number.size(); i++) {
        int my_block  = F0.block_number[i] - 1;
        int my_row    = F0.row[i] - 1;
        int my_column = F0.column[i] - 1;

        // calculate offset
        size_t off = 0;
        for (size_t j = 0; j < my_block; j++) {
            off += primal_block_dim[j] * primal_block_dim[j];
        }

        // populate relevant entry in c. note our definition of the problem has c = -F0
        c[off + my_row * primal_block_dim[my_block] + my_column] = -F0.value[i];
    }

    // constraint matrices
    for (size_t i = 0; i < Fi.size(); i++) {

        Fi_.push_back(Fi[i]);

        // add composite indices
        for (size_t j = 0; j < Fi[i].block_number.size(); j++) {

            int my_block  = Fi[i].block_number[j] - 1;
            int my_row    = Fi[i].row[j] - 1;
            int my_column = Fi[i].column[j] - 1;

            // calculate offset
            size_t off = 0;
            for (size_t k = 0; k < my_block; k++) {
                off += primal_block_dim[k] * primal_block_dim[k];
            }

            // composite index
            size_t id = off + my_row * primal_block_dim[my_block] + my_column;

            // add to matrix object
            Fi_[i].id.push_back(id);

        }

    }

    // primal block dimensions
    for (size_t i = 0; i < primal_block_dim.size(); i++) {
        primal_block_dim_.push_back(primal_block_dim[i]);
    }

    // primal solution vector (random guess on [-1:1])
    srand(0);
    double * x = (double*)malloc(n_primal_*sizeof(double));
    for (size_t i = 0; i < n_primal_; i++) {
        x[i] = 2.0 * ( (double)rand()/RAND_MAX - 1.0 );
    }

    // initialize sdp solver

    std::shared_ptr<SDPSolver> sdp;

    libsdp::SDPProgressMonitorFunction sdp_monitor;

    if ( options_.algorithm == SDPOptions::SDPAlgorithm::BPSDP ) {

        sdp = (std::shared_ptr<SDPSolver>)(new BPSDPSolver(n_primal_,n_dual_,options_));
        sdp_monitor = bpsdp_monitor;

    }else if ( options_.algorithm == SDPOptions::SDPAlgorithm::RRSDP ) {

        sdp = (std::shared_ptr<SDPSolver>)(new RRSDPSolver(n_primal_,n_dual_,options_));
        sdp_monitor = rrsdp_monitor;

    }

    // solve sdp
    sdp->solve(x,
               b.data(),
               c.data(),
               primal_block_dim_, 
               maxiter, 
               Au_callback, 
               ATu_callback, 
               sdp_monitor, 
               (void*)this);

    free(x);
}



SDPOptions options() {
    SDPOptions opt;
    return opt;
}


}

