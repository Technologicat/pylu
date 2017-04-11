# -*- coding: utf-8 -*-
#
"""Solve A x = b.

PyLU uses LU decomposition with partial pivoting (row swaps),
and requires only NumPy and Cython.

The main use case for PyLU (over numpy.linalg.solve) is solving many
small systems inside a nogil block in Cython code, without requiring
SciPy (for its cython_lapack module).

Python and Cython interfaces are provided. The API is designed
to be as simple to use as possible.
"""

from __future__ import absolute_import  # https://www.python.org/dev/peps/pep-0328/

__version__ = '0.1.1'

from .dgesv import *

