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

#ifndef SDPA_HELPER_H
#define SDPA_HELPER_H

#include<stdio.h>
#include<stdlib.h>

#include<vector>
#include<memory>

#include<sdp_solver.h>

namespace libsdp {

/// a constraint matrix in sparse SDPA format
struct SDPMatrix {
    SDPMatrix(){};
    std::vector<int> block_number;
    std::vector<int> row;
    std::vector<int> column;
    std::vector<double> value;

    /// composite index, accounting for row, column, block offset
    std::vector<int> id;
};

class SDPHelper{

  public:

    /// SDPHelper constructor
    SDPHelper(SDPOptions options, 
              std::vector<SDPMatrix> Fi,
              std::vector<int> primal_block_dim);

    /// SDPHelper destructor
    ~SDPHelper();

    /// solve the sdp problem and return the primal solution
    std::vector<double> solve(std::vector<double> b, 
                              int maxiter,
                              std::vector<int> primal_block_rank);

    /// evaluate Au
    void evaluate_Au(double * Au, double * u);

    /// evaluate ATu
    void evaluate_ATu(double * ATu, double * u);

    /// return the RRSDP penalty parameter, mu
    double get_mu();

    /// return the BPSDP dual z vector
    std::vector<double> get_z();

    /// return the BPSDP dual y vector or RRSDP lagrange multipliers
    std::vector<double> get_y();

    /// return the c vector
    std::vector<double> get_c() { return c_; }

    /// get ATu
    std::vector<double> get_ATu(std::vector<double> u);

    /// get Au
    std::vector<double> get_Au(std::vector<double> u);

  protected:

    /// the SDP solver
    std::shared_ptr<SDPSolver> sdp_;

    /// options for the SDP
    SDPOptions options_;

    /// the dimension of the primal vector (what SDPA calls the dual solution)
    long int n_primal_;

    /// the dimension of the dual vector (what SDPA calls the primal solution)
    long int n_dual_;

    /// the Fi matrices
    std::vector<SDPMatrix> Fi_;

    /// the FTi matrices
    std::vector<SDPMatrix> FTi_;

    /// list of block sizes
    std::vector<int> primal_block_dim_;

    /// list of block ranks (for low-rank RRSDP)
    std::vector<int> primal_block_rank_;

    /// the c vector (F0)
    std::vector<double> c_;

    /// the primal solution vector 
    std::vector<double> x_;

    /// progress monitor function
    libsdp::SDPProgressMonitorFunction sdp_monitor_;
};

}

#endif
