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

#include <stdio.h>

#include "lbfgs_helper.h"

#include <lbfgs.h>

namespace libsdp {

void lbfgs_error_check(int value) {

    //L-BFGS reaches convergence.
    //LBFGS_SUCCESS = 0,
    //LBFGS_CONVERGENCE = 0,
    //LBFGS_STOP,

    if (  value == 0 ) return;
    
    printf("\n");
    printf("    ==> WARNING <==\n");
    printf("\n");
    printf("    L-BFGS exited with an error:\n");
    printf("\n");

    if ( value == (int)LBFGS_ALREADY_MINIMIZED) 
        printf("        The initial variables already minimize the objective function.\n");
    if ( value == (int)LBFGSERR_UNKNOWNERROR) 
        printf("        Unknown error.\n");
    if ( value == (int)LBFGSERR_LOGICERROR) 
        printf("        Logic error.\n");
    if ( value == (int)LBFGSERR_OUTOFMEMORY) 
        printf("        Insufficient memory.\n");
    if ( value == (int)LBFGSERR_CANCELED) 
        printf("        The minimization process has been canceled.\n");
    if ( value == (int)LBFGSERR_INVALID_N) 
        printf("        Invalid number of variables specified.\n");
    if ( value == (int)LBFGSERR_INVALID_N_SSE) 
        printf("        Invalid number of variables (for SSE) specified.\n");
    if ( value == (int)LBFGSERR_INVALID_X_SSE) 
        printf("        The array x must be aligned to 16 (for SSE).\n");
    if ( value == (int)LBFGSERR_INVALID_EPSILON) 
        printf("        Invalid parameter lbfgs_parameter_t::epsilon specified.\n");
    if ( value == (int)LBFGSERR_INVALID_TESTPERIOD) 
        printf("        Invalid parameter lbfgs_parameter_t::past specified.\n");
    if ( value == (int)LBFGSERR_INVALID_DELTA) 
        printf("        Invalid parameter lbfgs_parameter_t::delta specified.\n");
    if ( value == (int)LBFGSERR_INVALID_LINESEARCH) 
        printf("        Invalid parameter lbfgs_parameter_t::linesearch specified.\n");
    if ( value == (int)LBFGSERR_INVALID_MINSTEP) 
        printf("        Invalid parameter lbfgs_parameter_t::max_step specified.\n");
    if ( value == (int)LBFGSERR_INVALID_MAXSTEP) 
        printf("        Invalid parameter lbfgs_parameter_t::max_step specified.\n");
    if ( value == (int)LBFGSERR_INVALID_FTOL) 
        printf("        Invalid parameter lbfgs_parameter_t::ftol specified.\n");
    if ( value == (int)LBFGSERR_INVALID_WOLFE) 
        printf("        Invalid parameter lbfgs_parameter_t::wolfe specified.\n");
    if ( value == (int)LBFGSERR_INVALID_GTOL) 
        printf("        Invalid parameter lbfgs_parameter_t::gtol specified.\n");
    if ( value == (int)LBFGSERR_INVALID_XTOL) 
        printf("        Invalid parameter lbfgs_parameter_t::xtol specified.\n");
    if ( value == (int)LBFGSERR_INVALID_MAXLINESEARCH)
        printf("        Invalid parameter lbfgs_parameter_t::max_linesearch specified.\n");
    if ( value == (int)LBFGSERR_INVALID_ORTHANTWISE)
        printf("        Invalid parameter lbfgs_parameter_t::orthantwise_c specified.\n");
    if ( value == (int)LBFGSERR_INVALID_ORTHANTWISE_START)
        printf("        Invalid parameter lbfgs_parameter_t::orthantwise_start specified.\n");
    if ( value == (int)LBFGSERR_INVALID_ORTHANTWISE_END)
        printf("        Invalid parameter lbfgs_parameter_t::orthantwise_end specified.\n");
    if ( value == (int)LBFGSERR_OUTOFINTERVAL)
        printf("        The line-search step went out of the interval of uncertainty.\n");
    if ( value == (int)LBFGSERR_INCORRECT_TMINMAX)
        printf("        A logic error occurred; alternatively, the interval of uncertainty became too small.\n");
    if ( value == (int)LBFGSERR_ROUNDING_ERROR)
        printf("        A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions.\n");
    if ( value == (int)LBFGSERR_MINIMUMSTEP)
        printf("        The line-search step became smaller than lbfgs_parameter_t::min_step.\n");
    if ( value == (int)LBFGSERR_MAXIMUMSTEP)
        printf("        The line-search step became larger than lbfgs_parameter_t::max_step.\n");
    if ( value == (int)LBFGSERR_MAXIMUMLINESEARCH)
        printf("        The line-search routine reaches the maximum number of evaluations.\n");
    if ( value == (int)LBFGSERR_MAXIMUMITERATION)
        printf("        The algorithm routine reaches the maximum number of iterations.\n");
    if ( value == (int)LBFGSERR_WIDTHTOOSMALL)
        printf("        Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol.\n");
    if ( value == (int)LBFGSERR_INVALIDPARAMETERS)
        printf("        A logic error (negative line-search step) occurred.\n");
    if ( value == (int)LBFGSERR_INCREASEGRADIENT)
        printf("        The current search direction increases the objective function value.\n");

    printf("\n");

    //exit(0);

}

}
