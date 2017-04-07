# -*- coding: utf-8 -*-
#
# Usage examples / unit test for PyLU.
#
# JJ 2017-04-07

from __future__ import division, print_function

import numpy as np

import pylu

def test():
    #########################################
    # set up test data
    #########################################

    A = np.array( [ [  4,  1,  0,  0,  2],
                    [  3,  3, -1, -1,  0],
                    [  0,  0,  1,  1,  1],
                    [ -2,  1, -2,  3,  7],
                    [  0,  0,  2,  2,  1] ], dtype=np.float64 )

    b = np.array(   [ 19,  6,  8, 31, 11], dtype=np.float64 )

    print( "Matrix A:" )
    print( A )
    print()
    print( "Load vector b:" )
    print( b )
    print()

    #########################################
    # solve  A x = b
    #########################################

    # simple API
    #
    x = pylu.solve( A, b )

    print( "Obtained solution of A x = b:" )
    print( x )
    print()
    print( "Check: A x =" )
    A_times_x = np.dot(A,x)
    print( A_times_x )
    print()
    print( "Residual A x - b:" )
    res = A_times_x - b
    print( res )
    print()
    assert np.allclose(res, 0.0)

    # two-step API (useful especially if many right-hand sides with the same A)
    #
    LU,p = pylu.lup_packed(A)
    x = pylu.solve_decomposed( LU, p, b )
    print( "Solution of A x = b using two-step prepare/solve procedure (packed LU format):" )
    print( x )
    print()
    res = np.dot(A,x) - b
    print( "Residual A x - b:" )
    print( res )
    print()
    assert np.allclose(res, 0.0)

    # banded matrix API
    #
    # Note: our A is not banded; this is just an example of how to use the API if it was.
    #
    # (It will still work, but provides no speed benefit in our test case.)
    #
    print( "Testing banded matrix support" )
    mincols,maxcols = pylu.find_bands(LU, tol=1e-15)
    x = pylu.solve_decomposed_banded( LU, p, mincols, maxcols, b )
    res = np.dot(A,x) - b
    print( "Residual A x - b:" )
    print( res )
    print()
    assert np.allclose(res, 0.0)

    #########################################
    # perform LU decomposition of A
    #########################################

    L,U,p = pylu.lup(A)
    print( "LU decomposition of A" )
    print()
    print( "L =" )
    print( L )
    print()
    print( "U =" )
    print( U )
    print()
    print( "p =" )
    print( p )
    print()
    print( "L U =" )
    print( np.dot(L,U) )
    print()
    print( "P A = " )
    print( A[p,:] )  # p is a vector for row permutation (a.k.a. partial pivoting)
    print()

    print( "Residual L U - P A:" )
    res = np.dot(L,U) - A[p,:]
    print( res )
    print()
    assert np.allclose(res, 0.0)


if __name__ == '__main__':
    test()

