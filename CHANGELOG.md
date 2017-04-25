## Changelog

### [v0.1.3]
 - support both Python 3.4 and 2.7

### [v0.1.2]
 - **bugfix**: `solve_decomposed` and `solve_decomposed_banded` now return proper `np.ndarray` objects (not memoryview slices)
 - setup.py is now Python 3 compatible
 - general tidiness:
   - added installation instructions to [README.md](README.md)
   - removed unused dependencies (oops)
   - fixed URL in setup.py

### [v0.1.1]
 - set zip_safe to False to better work with Cython (important for libs that depend on this one)

### [v0.1.0]
 - initial release as separate project

