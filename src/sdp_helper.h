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
    std::vector<int> constraint_number;
    std::vector<int> block_number;
    std::vector<int> row;
    std::vector<int> column;
    std::vector<double> value;
};

class SDPHelper{

  public:

    /// SDPHelper constructor
    SDPHelper(long int n_primal, long int n_dual, SDPOptions options);

    /// SDPHelper destructor
    ~SDPHelper();

    /// solve the sdp problem
    std::vector<double> solve(std::vector<double> x,
                              std::vector<double> b,
                              SDPMatrix F0,
                              std::vector<SDPMatrix> Fi,
                              std::vector<int> primal_block_dim,
                              int maxiter);
  protected:

    /// options for the SDP
    SDPOptions options_;

    /// the dimension of the primal vector (what SDPA calls the dual solution)
    long int n_primal_;

    /// the dimension of the dual vector (what SDPA calls the primal solution)
    long int n_dual_;

    /// the Fi matrices
    std::vector<SDPMatrix> Fi_;

    /// list of block sizes
    std::vector<int> primal_block_dim_;

};

}

#endif
