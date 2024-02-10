
#include "blas_helper.h"

#include <stdlib.h>

namespace libsdp {

/*!
 * This function scales a vector by a real scalar.
 *
 * \param length length of array
 * \param alpha  scale factor
 * \param vec    vector to scale
 * \param inc    how many places to skip to get to next element in vec
 *
 * \ingroup QT
 */
void C_DSCAL(size_t length, double alpha, double *vec, long int inc) {
    long int big_blocks = (long int)(length / INT_MAX);
    long int small_size = (long int)(length % INT_MAX);
    for (long int block = 0; block <= big_blocks; block++) {
        double *vec_s = &vec[static_cast<size_t>(block) * inc * INT_MAX];
        signed long int length_s = (block == big_blocks) ? small_size : INT_MAX;
        F_DSCAL(&length_s, &alpha, vec_s, &inc);
    }
}


/*!
 * This function returns the dot product of two vectors, x and y.
 *
 * \param length Number of elements in x and y.
 * \param x      A polong inter to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 * \param y      A polong inter to the beginning of the data in y.
 * \param inc_y  how many places to skip to get to next element in y
 *
 * @returns the dot product
 *
 */

double C_DDOT(size_t length, double *x, long int inc_x, double *y, long int inc_y) {
    if (length == 0) return 0.0;

    double reg = 0.0;

    long int big_blocks = (long int)(length / INT_MAX);
    long int small_size = (long int)(length % INT_MAX);
    for (long int block = 0; block <= big_blocks; block++) {
        double *x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        double *y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed long int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += F_DDOT(&length_s, x_s, &inc_x, y_s, &inc_y);
    }

    return reg;
}
/*!
 * This function returns the square of the norm of this vector.
 *
 * \param length Number of elements in x.
 * \param x      A polong inter to the beginning of the data in x.
 *               Must be of at least length (1+(N-1)*abs(inc_x).
 * \param inc_x  how many places to skip to get to next element in x
 *
 * @returns the norm squared product
 *
 */

double C_DNRM2(size_t length, double *x, long int inc_x) {
    if (length == 0) return 0.0;

    double reg = 0.0;

    long int big_blocks = (long int)(length / INT_MAX);
    long int small_size = (long int)(length % INT_MAX);
    for (long int block = 0; block <= big_blocks; block++) {
        double *x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        signed long int length_s = (block == big_blocks) ? small_size : INT_MAX;
        reg += F_DNRM2(&length_s, x_s, &inc_x);
    }

    return reg;
}

/*!
 * This function performs y = a * x + y.
 *
 * Steps every inc_x in x and every inc_y in y (normally both 1).
 *
 * \param length   length of arrays
 * \param a        scalar a to multiply vector x
 * \param x        vector x
 * \param inc_x    how many places to skip to get to next element in x
 * \param y        vector y
 * \param inc_y    how many places to skip to get to next element in y
 *
 */
void C_DAXPY(size_t length, double a, double *x, long int inc_x, double *y, long int inc_y) {
    long int big_blocks = (long int)(length / INT_MAX);
    long int small_size = (long int)(length % INT_MAX);
    for (long int block = 0; block <= big_blocks; block++) {
        double *x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        double *y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed long int length_s = (block == big_blocks) ? small_size : INT_MAX;
        F_DAXPY(&length_s, &a, x_s, &inc_x, y_s, &inc_y);
    }
}


/*!
 * This function copies x long into y.
 *
 * Steps every inc_x in x and every inc_y in y (normally both 1).
 *
 * \param length  = length of array
 * \param x       = vector x
 * \param inc_x   = how many places to skip to get to next element in x
 * \param y       = vector y
 * \param inc_y   = how many places to skip to get to next element in y
 *
 * \ingroup QT
 */
void C_DCOPY(size_t length, double *x, long int inc_x, double *y, long int inc_y) {
    long int big_blocks = (long int)(length / INT_MAX);
    long int small_size = (long int)(length % INT_MAX);
    for (long int block = 0; block <= big_blocks; block++) {
        double *x_s = &x[static_cast<size_t>(block) * inc_x * INT_MAX];
        double *y_s = &y[static_cast<size_t>(block) * inc_y * INT_MAX];
        signed long int length_s = (block == big_blocks) ? small_size : INT_MAX;
        F_DCOPY(&length_s, x_s, &inc_x, y_s, &inc_y);
    }
}

/**
 *  Diagonalize a real symmetric matrix
 */
void Diagonalize(long int N, double* A, double* W) {
    char JOBZ = 'V';
    char UPLO = 'U';
    long int LDA = N;
    long int LWORK = 3 * N - 1;
    double* WORK = (double*)malloc(LWORK * sizeof(double));
    long int INFO = 0;
    F_DSYEV(JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO);
    free(WORK);
}


}// end of namespaces
