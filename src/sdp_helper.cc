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

    // expor SDP options

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

    // export SDP solver

    py::class_<SDPHelper, std::shared_ptr<SDPHelper> >(m, "sdp_solver")
        .def(py::init<long int,long int, SDPOptions>())
        .def("solve", &SDPHelper::solve,
            "x"_a,
            "b"_a,
            "c"_a,
            "primal_block_dim"_a,
            "maxiter"_a,
            "evaluate_Au"_a,
            "evaluate_ATu"_a,
            "progress_monitor"_a);

}

PYBIND11_MODULE(pysdp, m) {
    m.doc() = "Python API of libsdp";
    export_SDPHelper(m);
}

/// SDPHelper constructor
SDPHelper::SDPHelper(long int n_primal, long int n_dual, SDPOptions options)
{

    if ( options.algorithm == SDPOptions::RRSDP ) {

        sdp_ = (std::shared_ptr<SDPSolver>)(new RRSDPSolver(n_primal,n_dual,options));

    }else if ( options.algorithm == SDPOptions::BPSDP ) {

        sdp_ = (std::shared_ptr<SDPSolver>)(new BPSDPSolver(n_primal,n_dual,options));

    }else {
        printf("\n");
        printf("    error: unknown SDP solver type\n");
        printf("\n");
        exit(1);
    }

}

SDPHelper::~SDPHelper()
{
}

std::vector<double> SDPHelper::solve(std::vector<double> x,
                                     std::vector<double> b,
                                     std::vector<double> c,
                                     std::vector<int> primal_block_dim,
                                     int maxiter,
                                     SDPCallbackFunction evaluate_Au,
                                     SDPCallbackFunction evaluate_ATu,
                                     SDPProgressMonitorFunction progress_monitor) {
    void * data;
    sdp_->solve(x.data(),b.data(),c.data(),primal_block_dim,maxiter,evaluate_Au,evaluate_ATu,progress_monitor,data);
    return x;

}

}
