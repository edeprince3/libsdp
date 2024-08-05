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

#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#include <cstdio>
#include <climits>
#include <cmath>

namespace libsdp{

#define F_DAXPY daxpy_
#define F_DCOPY dcopy_
#define F_DSCAL dscal_
#define F_DDOT  ddot_
#define F_DNRM2 dnrm2_
#define F_DGEMM dgemm_
#define F_DGEMV dgemv_
#define F_DSYEV dsyev_
#define F_DGEEV dgeev_

extern "C" {


extern void   F_DGEMM(char*, char*, long int*, long int*, long int*, double*, double*, long int*, double*, long int*, double*, double*, long int*);
extern void   F_DAXPY(long int *length, double *a, double *x, long int *inc_x, double *y, long int *inc_y);
extern void   F_DCOPY(long int *length, double *x, long int *inc_x, double *y, long int *inc_y);
extern void   F_DSCAL(long int *n, double *alpha, double *vec, long int *inc);
extern double F_DDOT(long int *n, double *x, long int *incx, double *y, long int *incy);
extern double F_DNRM2(long int *n, double *x, long int *incx);
extern void   F_DSYEV(char &JOBZ, char &UPLO, long int &N, double *A, long int &LDA, double *W, double *WORK,long int &LWORK, long int &INFO);
extern void   F_DGEEV(char &JOBVL, char &JOBVR, long int &N, double *A, long int &LDA, double *WR, double *WI, double *VL, long int &LDVL, double *VR, long int &LDVR, double *WORK, long int &LWORK, long int &INFO);

}

void C_DSCAL(size_t len, double alpha, double* vec, long int inc);
void C_DCOPY(size_t length, double* x, long int inc_x, double* y, long int inc_y);
void C_DAXPY(size_t length, double a, double* x, long int inc_x, double* y, long int inc_y);
double C_DDOT(size_t n, double* X, long int inc_x, double* Y, long int inc_y);
double C_DNRM2(size_t n, double* X, long int inc_x);

/**
 * diagonalize a real symmetric matrix
 */
void Diagonalize(long int N, double *A, double *W);
void Diagonalize_nonsym(long int N, double *A, double *W, double *VL, double *VR);


} // end of namespace

#endif
