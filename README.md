# PyLU

![top language](https://img.shields.io/github/languages/top/Technologicat/pylu)
![supported Python versions](https://img.shields.io/pypi/pyversions/pylu)
![supported implementations](https://img.shields.io/pypi/implementation/pylu)
![CI status](https://img.shields.io/github/actions/workflow/status/Technologicat/pylu/ci.yml?branch=master)

![version on PyPI](https://img.shields.io/pypi/v/pylu)
![PyPI package format](https://img.shields.io/pypi/format/pylu)
![dependency status](https://img.shields.io/librariesio/github/Technologicat/pylu)

![license](https://img.shields.io/pypi/l/pylu)
![open issues](https://img.shields.io/github/issues/Technologicat/pylu)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](http://makeapullrequest.com/)

Small nogil-compatible Cython-based solver for linear equation systems `A x = b`.

We use [semantic versioning](https://semver.org/).

For my stance on AI contributions, see the [collaboration guidelines](https://github.com/Technologicat/substrate-independent/blob/main/collaboration.md).


## Introduction

The algorithm is LU decomposition with partial pivoting (row swaps). The code requires only NumPy and Cython.

The main use case for PyLU (over `numpy.linalg.solve`) is solving many small systems inside a `nogil` block in Cython code, without requiring SciPy (for its `cython_lapack` module).

Python and Cython interfaces are provided. The API is designed to be as simple to use as possible.

The arrays are stored using the C memory layout.

A rudimentary banded solver is also provided, based on detecting the band structure (if any) from the initial full LU decomposition. For cases where `L` and `U` have small bandwidth, this makes the `O(n**2)` solve step run faster. The LU decomposition still costs `O(n**3)`, so this is useful only if the system is small, and the same matrix is needed for a large number of different RHS vectors. (This can be the case e.g. in integration of ODE systems with a constant-in-time mass matrix.)


## Examples

Basic usage:

```python
import numpy as np
import pylu

A = np.random.random( (5,5) )
b = np.random.random( 5 )

x = pylu.solve( A, b )
```

For a complete tour, see the [test suite](tests/test_dgesv.py).

The main item of interest, however, is the Cython API in [`dgesv.pxd`](pylu/dgesv.pxd). The main differences to the Python API are:

 - Function names end with `_c`.
 - Explicit sizes must be provided, since the arrays are accessed via raw pointers.
 - The result array `x` must be allocated by the caller, and passed in as an argument. See [`dgesv.pyx`](pylu/dgesv.pyx) for examples on how to do this in NumPy.


## Installation

### From PyPI

```bash
pip install pylu
```

### From source

```bash
git clone https://github.com/Technologicat/pylu.git
cd pylu
pip install .
```

For maximum performance on your machine, build with architecture-specific optimizations:

```bash
CFLAGS="-march=native" pip install --no-build-isolation .
```

Pre-built wheels from PyPI use generic `-O2` because `-march=native` bakes in the instruction set of the build machine — a wheel built with AVX-512 would crash on a CPU without it. Building from source avoids this and lets the compiler target your specific hardware.

### Development setup

PyLU uses [meson-python](https://meson-python.readthedocs.io/) as its build backend and [PDM](https://pdm-project.org/) for dependency management.

```bash
git clone https://github.com/Technologicat/pylu.git
cd pylu
pdm install
pip install --no-build-isolation -e .
```

The `--no-build-isolation` flag is required for editable installs with meson-python. Normally pip builds in a temporary isolated venv, but meson-python's editable mode needs the build dependencies (Cython, NumPy, meson) to remain available in the environment for on-import rebuilds.

**Note on editable installs:** meson-python editable installs rebuild the Cython extension on import via a redirect `.pth` file. After modifying `.pyx` or `.pxd` files, re-run `pip install --no-build-isolation -e .` to rebuild. Alternatively, use a non-editable install (`pip install .`) and reinstall after changes.


## Dependencies

- [NumPy](http://www.numpy.org) ≥ 1.25
- [Cython](http://www.cython.org) ≥ 3.0 (build-time only)

Requires Python ≥ 3.11.


## License

[BSD](LICENSE.md). Copyright 2016–2026 Juha Jeronen, University of Jyväskylä, and JAMK University of Applied Sciences.


#### Acknowledgement

This work was financially supported by the Jenny and Antti Wihuri Foundation.
