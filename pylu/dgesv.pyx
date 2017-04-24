# -*- coding: utf-8 -*-
#
# A simple partial pivoting (row swap) LU decomposition based solver for linear equation systems.
# Dense matrices, real number domain. Support for banded matrices available.
#
# This implementation is pure Cython; the only dependency is NumPy, needed for the Python interface demonstrating usage with NumPy arrays.
#
# The point of this module is to be able to solve reasonably sized dense linear equation systems inside nogil blocks without requiring external dependencies.
#
#
# This is a Cython version of a C++ implementation originally made for teaching exercise classes for the course
# TIEA381 Numeeriset menetelmät (Numerical methods) at University of Jyväskylä, fall 2010.
#
# The LU decomposition algorithm was given in the lecture notes, pp. 27-28; it is one of the standard algorithms.
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

from __future__ import division

import numpy as np

cimport dgesv  # Cython interface and implementation

#################################################
# Python interface for demonstration and testing
#################################################

# Python asserts are *slow*, hence disabled for now.


# fire-and-forget simple version

def solve( double[:,::1] A, double[::1] b ):
    """def solve( double[:,::1] A, double[::1] b ):

    Solve  A x = b.

    Parameters:
        A [in]  = rank-2 np.array of double. Must be square.
        b [in]  = rank-1 np.array of double. Load vector, length must match the number of rows in A.

    Returns:
        x       = rank-1 np.array of double. Solution vector.

    Raises RuntimeError if LU decomposition fails.

    """
#    assert( A.shape[0] == A.shape[1], "Matrix A must be square" )
#    assert( A.shape[0] == b.shape[0], "Length of load vector b must match the number of rows in A" )

    cdef double[::1] x = np.empty( (b.shape[0],), dtype=np.float64, order="C" )

#    cdef int i
    cdef int n = A.shape[0]
    cdef int success
    with nogil:
#        # For performance measurement. Note that for reliable results benchmarking the C code only (a realistic use case if used from inside a Cython-accelerated solver),
#        # we must loop here and not at the calling end, because a large majority of the execution time is taken up by data conversion from Python to C and back (and Python asserts, if enabled).
#        for i in range(1000000):
#            success = solve_c( &A[0,0], &b[0], &x[0], n )
        success = solve_c( &A[0,0], &b[0], &x[0], n )
    if not success:
        raise RuntimeError("Solution of A x = b failed; matrix maybe close to singular?")

    return np.asanyarray(x)

# ------------------------------------------------------------------------------------------------------------------------------------------

# separate LU factorization and solving (for solving with many right-hand sides for the same matrix)

def lup( double[:,::1] A ):
    """def lup( double[:,::1] A ):

    Perform LU decomposition of a square matrix A with partial pivoting (row swaps).

    Constructs L, U, p such that
    
      P A = L U

    where the permutation matrix P permutes rows. In terms of the permutation vector p
    that is actually returned by this function, we can write (in mixed matrix and NumPy notation):

      P A = A[p,:]


    Unlike lup_packed(), the matrices L and U returned by lup() are meant to be human-readable.


    Parameters:
        A = rank-2 np.array of doubles. Square matrix to be decomposed.

    Returns:
        tuple (L,U,p)

        L = rank-2 np.array of double. Lower triangular matrix with ones on the diagonal.
        U = rank-2 np.array of double. Upper triangular matrix.
        p = rank-1 np.array of intc (C-compatible int). Row permutation vector.

    Raises RuntimeError if LU decomposition fails.

    """
    assert( A.shape[0] == A.shape[1], "Matrix A must be square" )

    cdef int[::1] p
    cdef double[:,::1] LU
    LU,p = lup_packed(A)

    # Extract L and U from packed format.
    #
    cdef double[:,::1] L = np.zeros_like(A)
    cdef double[:,::1] U = np.zeros_like(A)
    cdef int n = A.shape[0]
    cdef int i
    for i in range(n):
        L[i,:i] = LU[i,:i]
        L[i,i]  = 1.0  # diagonal of L is ones
        U[i,i:] = LU[i,i:]

    return (np.asanyarray(L), np.asanyarray(U), np.asanyarray(p))


def lup_packed( double[:,::1] A ):
    """def lup_packed( double[:,::1] A ):

    Like lup(), but return the LU decomposition in packed storage format.

    This is the raw data format; the purpose is to prepare to solve many linear equation systems with the same matrix.
    See solve_decomposed().

    Parameters:
        A = rank-2 np.array of doubles. Square matrix to be decomposed.

    Returns:
        tuple (LU,p)

        LU = rank-2 np.array of double. Packed LU decomposition.
        p  = rank-1 np.array of intc. Row permutation vector.

    Raises RuntimeError if LU decomposition fails.

    """
    assert( A.shape[0] == A.shape[1], "Matrix A must be square" )

    cdef int n            = A.shape[0]
    cdef double[:,::1] LU = A.copy()
    cdef int[::1] p       = np.empty( (n,), dtype=np.intc, order="C" )  # intc for C-compatible int; http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in

    cdef int success
    with nogil:
        success = decompose_lu_inplace_c( &LU[0,0], &p[0], n )
    if not success:
        raise RuntimeError("Failed to LU decompose A")

    return (np.asanyarray(LU), np.asanyarray(p))


def solve_decomposed( double[:,::1] LU, int[::1] p, double[::1] b ):
    """def solve_decomposed( double[:,::1] LU, int[::1] p, double[::1] b ):

    Solve  A x = b  using a previously prepared LU decomposition of A (see lup_packed()).

    Parameters:
        LU [in]  = rank-2 np.array of double. Packed LU decomposition of A, as output by lup_packed().
        p  [in]  = rank-1 np.array of double. Permutation vector, as output by lup_packed().
        b  [in]  = rank-1 np.array of double. Load vector, length must match the number of rows in A.

    Returns:
        x        = rank-1 np.array of double. Solution vector.
    """
#    assert( LU.shape[0] == LU.shape[1], "Matrix LU must be square" )
#    assert( LU.shape[0] == np.shape(p)[0],  "Length of permutation vector b must match the number of rows in LU" )
#    assert( LU.shape[0] == b.shape[0],  "Length of load vector b must match the number of rows in LU" )

    cdef double[::1] x = np.empty( (b.shape[0],), dtype=np.float64, order="C" )

#    cdef int i
    cdef int n = LU.shape[0]
    with nogil:
#        # Performance measurement.
#        for i in range(1000000):
#            solve_decomposed_c( &LU[0,0], &p[0], &b[0], &x[0], n )
        solve_decomposed_c( &LU[0,0], &p[0], &b[0], &x[0], n )

    return np.asanyarray(x)

# ------------------------------------------------------------------------------------------------------------------------------------------

# version optimized for banded L and U

def find_bands( double[:,::1] LU, double tol ):
    """def find_bands( double[:,::1] LU, double tol ):

    Find bands of nonzeroes in L and U.

    This saves a lot of flops when solving, if the bandwidths of L and U are small (especially significant
    if solving a large number of equation systems using the same matrix).

    The band for L extends from the first nonzero element on each row to the diagonal.
    The band for U extends from the diagonal to the last nonzero element.

    Parameters:
        LU      [in]  = rank-2 np.array of double. Packed LU decomposition of A, as output by lup_packed().

    Returns:
        tuple (mincols, maxcols)

        mincols = rank-1 np.array of intc. mincols[i] gives the column index of the first nonzero element on row i in L.
        maxcols = rank-1 np.array of intc. maxcols[i] gives the column index of the last  nonzero element on row i in U.

    """

    assert( LU.shape[0] == LU.shape[1], "Matrix LU must be square" )

    cdef int n = LU.shape[0]
    cdef int[::1] mincols = np.empty( (n,), dtype=np.intc, order="C" )
    cdef int[::1] maxcols = np.empty( (n,), dtype=np.intc, order="C" )

    with nogil:
        find_bands_c( &LU[0,0], &mincols[0], &maxcols[0], n, tol )

    return (np.asanyarray(mincols), np.asanyarray(maxcols))


def solve_decomposed_banded( double[:,::1] LU, int[::1] p, int[::1] mincols, int[::1] maxcols, double[::1] b ):
    """def solve_decomposed_banded( double[:,::1] LU, int[::1] p, int[::1] mincols, int[::1] maxcols, double[::1] b ):

    Solve  A x = b  using a previously prepared LU decomposition of A (see lup_packed()).

    Parameters:
        LU      [in]  = rank-2 np.array of double. Packed LU decomposition of A, as output by lup_packed().
        p       [in]  = rank-1 np.array of double. Permutation vector, as output by lup_packed().
        mincols [in]  = rank-1 np.array of intc. Band information for L, as output by find_bands().
        maxcols [in]  = rank-1 np.array of intc. Band information for U, as output by find_bands().
        b       [in]  = rank-1 np.array of double. Load vector, length must match the number of rows in A.

    Returns:
        x             = rank-1 np.array of double. Solution vector.
    """
#    assert( LU.shape[0] == LU.shape[1],       "Matrix LU must be square" )
#    assert( LU.shape[0] == np.shape(p)[0],        "Length of permutation vector b must match the number of rows in LU" )
#    assert( LU.shape[0] == np.shape(mincols)[0],  "Length of mincols vector must match the number of rows in LU" )
#    assert( LU.shape[0] == np.shape(maxcols)[0],  "Length of maxcols vector must match the number of rows in LU" )
#    assert( LU.shape[0] == b.shape[0],        "Length of load vector b must match the number of rows in LU" )

    cdef double[::1] x = np.empty( (b.shape[0],), dtype=np.float64, order="C" )

#    cdef int i
    cdef int n = LU.shape[0]
    with nogil:
#        # Performance measurement.
#        for i in range(1000000):
#            solve_decomposed_banded_c( &LU[0,0], &p[0], &mincols[0], &maxcols[0], &b[0], &x[0], n )
        solve_decomposed_banded_c( &LU[0,0], &p[0], &mincols[0], &maxcols[0], &b[0], &x[0], n )

    return np.asanyarray(x)

