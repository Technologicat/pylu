# -*- coding: utf-8 -*-
#
# Cython interface and implementation for dgesv.pyx.
#
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

# use fast math functions from <math.h>, available via Cython
from libc.math cimport fabs as c_abs

from libc.stdlib cimport malloc, free

cimport cython  # Cython compiler directives (e.g. boundscheck)


DEF DIAG_ELEMENT_ABS_TOL = 1e-15


# LU decompose an  n x n  matrix in-place, using partial pivoting (row swaps).
#
# Produces the matrices L and U such that  P A = L U,  where P is a row-swapping permutation matrix.
#
# A (in/out) - rank-2 array, size (n,n), C storage order; a square matrix to be LU decomposed
# p (out)    - rank-1 array, size (n); will be filled with the permutation vector used in partial pivoting.
# n (in)     - number of rows in A
#
# Return value:
#    True  - success
#    False - failed, absolute value of a diagonal element too small even after pivoting
#
# The matrix A is modified in-place. If the operation fails, A will be left in a corrupted state.
#
# The storage format is packed so that the diagonal elements of L are implicitly 1 (not stored).
# The diagonal of the decomposed A stores the diagonal elements of U.
#
cdef inline int decompose_lu_inplace_c( double* A, int* p, int n ) nogil:
    cdef int i, j, k, r, s

    # Initialize permutation vector to range(n)
    for k in range(n):
        p[k] = k

    cdef double mag, max_mag
    for k in range(n):
        # find row  r  such that  |a_rk| = max( |a_sk|, s = k, k+1, ..., n-1 )   (note 0-based indexing, last element is n-1)
        max_mag = -1.0  # any invalid value
        r = -1
        for s in range(k, n):
            mag = c_abs( A[s*n + k] )
            if mag > max_mag:
                max_mag = mag
                r = s
        if r == -1:  # cannot happen
            return False

        # swap elements k and r of p
        p[k], p[r] = p[r], p[k]

        # physically swap also the corresponding rows of A.
        #
        # This may seem a silly way to do it (why not indirect via p?), but it makes future accesses faster,
        # because then we won't need to use the permutation vector p when accessing A or LU.
        #
        # (It is still needed for accessing the load vector when solving the actual equation system.)
        #
        # See commit d10b724ef72d94d96847293bf5e20dd0e6ec5a14 in the repo for a version that *does*
        # indirect via p.
        #
        for j in range(n):
            A[k*n + j], A[r*n + j] = A[r*n + j], A[k*n + j]

        if c_abs(A[k*n + k]) <= DIAG_ELEMENT_ABS_TOL:
            return False

        for i in range(k+1, n):
            A[i*n + k] /= A[k*n + k]
            for j in range(k+1, n):
                A[i*n + j] -= A[i*n + k] * A[k*n + j]

#    # DEBUG
#    with gil:
#        for i in range(n):
#            print p[i]

    return True


# Solve a linear equation system after the matrix has been LU-decomposed.
#
# A (in)  - rank-2 array, size (n,n), C storage order; packed LU decomposition from decompose_lu_inplace_c()
# p (in)  - rank-1 array, size (n); permutation vector from decompose_lu_inplace_c()
# b (in)  - rank-1 array, size (n); load vector (right-hand side of A x = b)
# x (out) - rank-1 array, size (n); solution for P A x = P b
# n (in)  - number of rows in A
#
# The equation to be solved is
#
# A x = b
#
# in the form
#
# P A x = P b
#
# where P is a permutation matrix (permuting rows) and P A = L U.
#
# Hence
#
# L U x = P b
#
# This equation is equivalent with the system
#
# L y = P b
# U x = y
#
# We first solve L y = P b and then U x = y.
#
cdef inline void solve_decomposed_c( double* LU, int* p, double* b, double* x, int n ) nogil:
    cdef int i, j

    # Solve  L y = P b  by forward substitution
    #
    for i in range(n):
        x[i] = b[p[i]]  # formally, (P b)[i]
        for j in range(i):
            x[i] -= LU[ i*n + j ] * x[j]
    # Now the array "x" contains the solution of the first equation, i.e. y.

    # Solve  U x = y  by backward substitution
    #
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= LU[ i*n + j ] * x[j]
        x[i] /= LU[ i*n + i ]
    # Now the array "x" contains the solution of the second equation, i.e. x.


# "Sparsified" version, which takes two additional arguments:
#
# mincols[i] gives the column index of the first nonzero element on row i in L
# maxcols[i] gives the column index of the last nonzero element on row i in U
#
# The loop for L ranges from [mincols[i], diagonal), and the loop for U ranges from [diagonal, maxcols[i]].
# This avoids performing no-ops with array elements which are known to be zero.
#
# This saves a lot of flops if the bandwidths of L and U are small (especially significant if solving a large number of
# equation systems using the same matrix).
#
cdef inline void solve_decomposed_banded_c( double* LU, int* p, int* mincols, int* maxcols, double* b, double* x, int n ) nogil:
    cdef int i, j

    # Solve  L y = P b  by forward substitution
    #
    for i in range(n):
        x[i] = b[p[i]]  # formally, (P b)[i]
        for j in range(mincols[i], i):
            x[i] -= LU[ i*n + j ] * x[j]
    # Now the array "x" contains the solution of the first equation, i.e. y.

    # Solve  U x = y  by backward substitution
    #
    for i in range(n-1, -1, -1):
        for j in range(i+1, maxcols[i]+1):
            x[i] -= LU[ i*n + j ] * x[j]
        x[i] /= LU[ i*n + i ]
    # Now the array "x" contains the solution of the second equation, i.e. x.


# Generate mincols and maxcols for solve_decomposed_c_banded().
#
# tol sets the tolerance for the nonzero check.
#
# Note that this needs to be done only once for each LU decomposed matrix, so the sequence is:
#
#   // prepare (only once per matrix)
#   decompose_lu_inplace_c(...)
#   find_bands_c(...)   // if you know your L and U matrices will have small bandwidth
#
#   // solve (N times, one for each right-hand side)
#   solve_decomposed_c_banded(...)
#
cdef inline void find_bands_c( double* LU, int* mincols, int* maxcols, int n, double tol ) nogil:
    cdef int i, j

    for i in range(n):
        # For an invertible matrix, the diagonals of L and U are always nonzero
        mincols[i] = i
        maxcols[i] = i

        # L
        #
        for j in range(i):
            if c_abs( LU[ i*n + j ] ) > tol:
                mincols[i] = j
                break

        # U
        #
        for j in range(n-1, i, -1):
            if c_abs( LU[ i*n + j ] ) > tol:
                maxcols[i] = j
                break


# Solve a general linear equation system A x = b.
#
# A (in)  - rank-2 array, size (n,n), C storage order
# b (in)  - rank-1 array, size (n); load vector (right-hand side of A x = b)
# x (out) - rank-1 array, size (n); solution for A x = b
# n (in)  - number of rows in A
#
cdef inline int solve_c( double* A, double* b, double* x, int n ) nogil:
    cdef int* p     = <int*>(    malloc( n   * sizeof(int)    ) )
    cdef double* LU = <double*>( malloc( n*n * sizeof(double) ) )

    # Make a copy of A, since the decomposer will overwrite its input
    cdef int i, j
    for i in range(n):
        for j in range(n):
            LU[ i*n + j ] = A[ i*n + j ]

    if not decompose_lu_inplace_c( LU, p, n ):
        return False
    solve_decomposed_c( LU, p, b, x, n )

    free(p)
    p = NULL

    free(LU)
    LU = NULL

    return True

