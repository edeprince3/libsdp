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

#ifndef BPSDP_SOLVER_H
#define BPSDP_SOLVER_H

#include "sdp_solver.h"

namespace libsdp {

class BPSDPSolver: public SDPSolver{

  public:

    /// BPSDPSolver constructor
    BPSDPSolver(long int n_primal, long int n_dual,SDPOptions options);

    /// BPSDPSolver destructor
    ~BPSDPSolver();

    /// solve the sdp problem
    void solve(double * x, 
               double * b, 
               double * c,
               std::vector<int> primal_block_dim,
               int maxiter,
               SDPCallbackFunction evaluate_Au,
               SDPCallbackFunction evaluate_ATu,
               SDPProgressMonitorFunction progress_monitor,
               void * data);

    void evaluate_AATu(double * AATu,double * u);

  protected:

    /// a vector the size of the primal vector for containing ATu during the CG solve
    double * ATu_;

    /// the right-hand side of AATy = A(c - z) + mu(b - Ax) 
    double * cg_rhs_;

    /// update the primal (x) and dual (z) solutions
    void Update_xz(double * x, 
                   double * c, 
                   std::vector<int> primal_block_dim, 
                   SDPCallbackFunction evaluate_ATu,
                   void * data);

};

}

#endif
