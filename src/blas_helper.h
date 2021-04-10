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

#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#include <cstdio>
#include <climits>
#include <cmath>

namespace libsdp{

void C_DSCAL(size_t len, double alpha, double* vec, int inc);
void C_DCOPY(size_t length, double* x, int inc_x, double* y, int inc_y);
void C_DAXPY(size_t length, double a, double* x, int inc_x, double* y, int inc_y);
double C_DDOT(size_t n, double* X, int inc_x, double* Y, int inc_y);
double C_DNRM2(size_t n, double* X, int inc_x);

} // end of namespace

#endif
