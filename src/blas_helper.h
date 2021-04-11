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

#define F_DAXPY daxpy_
#define F_DCOPY dcopy_
#define F_DSCAL dscal_
#define F_DDOT ddot_
#define F_DNRM2 dnrm2_
#define F_DGEMM dgemm_
#define F_DGEMV dgemv_
#define F_DSYEV dsyev_

extern "C" {


extern void F_DGEMM(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
extern void F_DAXPY(int *length, double *a, double *x, int *inc_x, double *y, int *inc_y);
extern void F_DCOPY(int *length, double *x, int *inc_x, double *y, int *inc_y);
extern void F_DSCAL(int *n, double *alpha, double *vec, int *inc);
extern double F_DDOT(int *n, double *x, int *incx, double *y, int *incy);
extern double F_DNRM2(int *n, double *x, int *incx);
extern void F_DSYEV(char &JOBZ, char &UPLO, int &N, double *A, int &LDA, double *W, double *WORK,int &LWORK, int &INFO);

}

void C_DSCAL(size_t len, double alpha, double* vec, int inc);
void C_DCOPY(size_t length, double* x, int inc_x, double* y, int inc_y);
void C_DAXPY(size_t length, double a, double* x, int inc_x, double* y, int inc_y);
double C_DDOT(size_t n, double* X, int inc_x, double* Y, int inc_y);
double C_DNRM2(size_t n, double* X, int inc_x);

/**
 * diagonalize a real symmetric matrix
 */
void Diagonalize(long int N, double *A, double *W);


} // end of namespace

#endif
