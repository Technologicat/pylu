# PyLU Build System Migration Brief

## For CC implementation. Author: Juha Jeronen + Claude (claude.ai design session).

**Date:** 2026-03-31
**Target:** PyLU v1.0.0 — modernized build, Cython 3.x, Python 3.10+
**Repo:** https://github.com/Technologicat/pylu
**Current version:** v0.1.3 (April 2017)

---

## 1. Project Overview

PyLU is a small Cython library providing LU decomposition with partial pivoting
for solving `A x = b`. Its raison d'être is calling `dgesv` inside `nogil` blocks
without depending on SciPy. It exposes both Python and Cython (`cimport`) APIs.

### File inventory (verified)

```
pylu/
├── pylu/
│   ├── __init__.py      # Package init, version, re-exports via `from .dgesv import *`
│   ├── dgesv.pyx        # Python-facing wrappers (~265 lines, 6 public functions)
│   └── dgesv.pxd        # C-level inline implementations (~240 lines, 5 _c functions)
├── test/
│   └── pylu_test.py     # Combined usage example / test script (not pytest)
├── setup.py             # Old-style setuptools + cythonize(), math flags, libm linking
├── README.md
├── CHANGELOG.md
├── LICENSE.md
└── .gitignore
```

### Dependencies

- **Build-time:** Cython, NumPy (for headers)
- **Runtime:** NumPy
- **System:** libm (math library; linked via `libc.math cimport fabs` in `.pxd`)
- No SciPy, no LAPACK, no external C libs beyond libc/libm.

### Critical constraint

**The `.pxd` file must be installed alongside the compiled `.so`**, so that
downstream projects (pydgq, python-wlsqm) can `cimport pylu.dgesv`. This
is the single most important thing to verify after migration. Cython searches
`sys.path` for `.pxd` files during compilation, so a standard pip install into
site-packages works — but only if the `.pxd` is actually included in the wheel.

---

## 2. Build Backend Decision: meson-python

Use **meson-python** as the build backend. Rationale:

- **Consistency:** All three numerics projects (PyLU → pydgq → python-wlsqm) will
  use the same backend. python-wlsqm needs meson for OpenMP; using it everywhere
  avoids maintaining two build system patterns.
- **Ecosystem alignment:** SciPy, scikit-learn, and much of the scientific Python
  ecosystem now use meson. Cython's own docs recommend it for new projects.
- **Native Cython support:** Meson handles `.pyx` → `.c` → `.so` natively via
  `py.extension_module()`. No need for `cythonize()` calls.
- **Cross-platform:** Meson abstracts compiler detection and flags across
  Linux, macOS, and Windows.

### Known meson/Cython rough edges (and why they don't bite us here)

- **`.pxd` dependency tracking** (meson issue #9049): Changes to `.pxd` files
  don't always trigger rebuilds during development. For PyLU this is irrelevant —
  there's only one `.pyx`/`.pxd` pair and they change together. For pydgq/wlsqm
  it matters more, but a clean build after `.pxd` changes is a simple workaround.
- **Cross-package cimport at build time**: When building pydgq, Cython needs to
  find PyLU's `.pxd` on `sys.path`. This works automatically if PyLU is pip-installed
  into the build environment. Document this as a build dependency.

---

## 3. File-by-File Migration Plan

### 3.1. DELETE: `setup.py`

Remove entirely after migration is complete. All build config moves to
`pyproject.toml` and `meson.build`.

### 3.2. CREATE: `pyproject.toml`

```toml
[build-system]
requires = ["meson-python>=0.16", "Cython>=3.0", "numpy>=1.24"]
build-backend = "mesonpy"

[project]
name = "pylu"
version = "1.0.0"
description = "Small nogil-compatible linear equation system solver"
readme = "README.md"
license = "BSD-2-Clause"
requires-python = ">=3.10"
authors = [
    {name = "Juha Jeronen", email = "juha.jeronen@jamk.fi"},
]
keywords = [
    "numerical", "linear-equations", "solver",
    "lu-decomposition", "cython", "numpy",
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
]
dependencies = ["numpy>=1.24"]

[project.urls]
Homepage = "https://github.com/Technologicat/pylu"
Repository = "https://github.com/Technologicat/pylu"
Issues = "https://github.com/Technologicat/pylu/issues"
Changelog = "https://github.com/Technologicat/pylu/blob/master/CHANGELOG.md"
```

**Notes:**
- Version bump to 1.0.0: breaking change (dropped Python 2, new build system).
- `Operating System :: OS Independent` replaces `POSIX :: Linux` — we're targeting
  all three platforms via cibuildwheel now.
- Email: use your current preferred email, not the JYU one.
- `requires-python = ">=3.10"` matches your other modernized projects.

### 3.3. CREATE: `meson.build` (top-level)

```meson
project(
    'pylu',
    'cython', 'c',
    version: '1.0.0',
    license: 'BSD-2-Clause',
    meson_version: '>=1.1.0',
    default_options: [
        'buildtype=release',
        'c_std=c11',
        'cython_language=c',
    ],
)

py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')

# NumPy include directory (needed for Cython compilation against NumPy)
numpy_dep = dependency('numpy')

subdir('pylu')
```

**Notes:**
- `pure: false` tells meson-python this is a platform-specific (compiled) package.
- `numpy` dependency is detected via `dependency()` which uses NumPy's pkg-config
  or fallback detection. If this causes trouble on some platforms, alternative is:
  ```meson
  numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'],
                          check: true).stdout().strip()
  numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))
  ```

### 3.4. CREATE: `pylu/meson.build`

```meson
# libm dependency (dgesv.pxd uses libc.math fabs)
# On Linux this links -lm; on macOS/Windows it's a no-op (math is in libc/MSVCRT).
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# Cython extension module
py.extension_module(
    'dgesv',
    'dgesv.pyx',
    dependencies: [numpy_dep, m_dep],
    install: true,
    subdir: 'pylu',
)

# Install Python files
py.install_sources(
    '__init__.py',
    subdir: 'pylu',
)

# CRITICAL: Install .pxd for downstream cimport
# Without this, pydgq and python-wlsqm cannot cimport pylu.dgesv
py.install_sources(
    'dgesv.pxd',
    subdir: 'pylu',
)
```

**This is the most important file in the migration.** The `.pxd` install line
is what makes downstream `cimport pylu.dgesv` work.

**On compiler flags:** The old `setup.py` used `-march=native -msse -msse2 -mfma
-mfpmath=sse`. These must NOT go into the default build — they make wheels
non-portable. Meson's `buildtype=release` gives `-O2` which is correct for
distributed wheels. Users building from source who want architecture tuning
can set `CFLAGS="-march=native"` before building — that one flag is
sufficient. `-msse`/`-msse2`/`-mfpmath=sse` are no-ops on x86_64 (SSE2 is
part of the x86_64 baseline; GCC's default `-mfpmath` is already `sse`),
and `-mfma` is either implied by `-march=native` on Haswell+ (2013 and
later) or not supported by the CPU at all. Do not hardcode any of these.
Verify after migration by:
1. `pip install .` (or `pip install dist/pylu-1.0.0-*.whl`)
2. Check that `dgesv.pxd` exists in the installed package:
   `python -c "import pylu; import pathlib; print(list(pathlib.Path(pylu.__file__).parent.glob('*.pxd')))"`
3. In a separate directory, create a test `.pyx` file with `cimport pylu.dgesv`
   and verify Cython can compile it.

### 3.5. MODIFY: `pylu/__init__.py`

Current state (verified):
```python
from __future__ import absolute_import
__version__ = '0.1.3'
from .dgesv import *
```

Modernize to:
```python
"""PyLU — Small nogil-compatible linear equation system solver.

LU decomposition with partial pivoting for solving A x = b.
Both Python and Cython (cimport) interfaces are provided.
"""

from pylu.dgesv import (solve,
                         lup, lup_packed,
                         solve_decomposed,
                         find_bands, solve_decomposed_banded)

__version__ = "1.0.0"
```

**Notes:**
- Remove the `from __future__` import.
- The `from .dgesv import *` works but explicit imports are preferable for
  a public API. Either form is acceptable — Juha's call.
- The existing docstring in `__init__.py` is good; keep or update it.

### 3.6. MODIFY: `pylu/dgesv.pyx`

This is where the Cython 3.x audit matters. Changes needed:

#### 3.6.1. Verify/update Cython directive header

The file already has (verified from source):
```python
# cython: wraparound = False
# cython: boundscheck = False
# cython: cdivision = True
```

Add `language_level` to make Cython 3 behavior explicit:
```python
# cython: language_level=3
# cython: wraparound = False
# cython: boundscheck = False
# cython: cdivision = True
```

#### 3.6.2. Fix relative cimport (BREAKING in Cython 3)

Line 28 currently reads:
```python
cimport dgesv  # Cython interface and implementation
```

In Cython 3 with `language_level=3`, bare `cimport dgesv` is an absolute import,
meaning it looks for a top-level `dgesv` module — not the `pylu.dgesv` sitting
right next to this file. Change to:
```python
from . cimport dgesv  # Cython interface and implementation
```

**This is the single most likely Cython 3 breakage in this project.**

#### 3.6.3. Remove `__future__` imports

Delete all lines like:
```python
from __future__ import division, print_function, absolute_import
```

#### 3.6.4. Remove Python 2 compatibility

Delete any `try: cPickle ... except: pickle` patterns, `xrange`, `unicode`
references, or similar. (Not present in `dgesv.pyx` — verified.)

#### 3.6.5. Cython 3.x audit checklist

Review the following in `dgesv.pyx` and `dgesv.pxd`:

| Item | Status | Action |
|------|--------|--------|
| `cdivision` | ✅ Already set at module level in both files | None needed |
| `boundscheck` | ✅ Already `False` at module level in both files | None needed |
| `wraparound` | ✅ Already `False` at module level in both files | None needed |
| Relative `cimport` | ⚠️ Bare `cimport dgesv` on `.pyx` line 28 | Change to `from . cimport dgesv` — **critical** |
| `__future__` imports | ⚠️ `from __future__ import division` in `.pyx` | Remove |
| `DEF` | ⚠️ `DEF DIAG_ELEMENT_ABS_TOL = 1e-15` in `.pxd` line 24 | Replace with `cdef extern from *` + `#define` — see below |
| `cdef` exception specs | ⚠️ All five `_c` functions lack exception specs | Add `noexcept` to all — see below |
| `np.ndarray` typed args | ✅ Already uses memoryviews in `.pyx` | None needed |
| Raw pointer casts | ✅ Uses `&arr[0,0]` on memoryviews in `.pyx` | Correct pattern, keep as-is |
| `__init__.py` | ⚠️ Has `from __future__ import absolute_import` | Remove; `from .dgesv import *` is fine to keep |

#### Details on the `DEF` replacement

Line 24 of `dgesv.pxd`:
```python
DEF DIAG_ELEMENT_ABS_TOL = 1e-15
```

`DEF` is deprecated in Cython 3. Replace with a C-level `#define` via verbatim
C injection, which preserves the named constant, works in `nogil`, and has zero
runtime cost:

```python
cdef extern from *:
    """
    #define DIAG_ELEMENT_ABS_TOL 1e-15
    """
    double DIAG_ELEMENT_ABS_TOL
```

The call site (line 80) does not change.

#### Details on exception specs

All five `_c` functions in `dgesv.pxd` are `cdef inline ... nogil` and never
raise Python exceptions. They communicate errors via return codes (True/False).
In Cython 3, `cdef` functions without an explicit exception spec get implicit
exception checking, which is unnecessary overhead for `nogil` functions that
can't raise. Add `noexcept`:

```python
# Before:
cdef inline int decompose_lu_inplace_c( double* A, int* p, int n ) nogil:
cdef inline void solve_decomposed_c( double* LU, int* p, double* b, double* x, int n ) nogil:
cdef inline void solve_decomposed_banded_c( double* LU, int* p, int* mincols, int* maxcols, double* b, double* x, int n ) nogil:
cdef inline void find_bands_c( double* LU, int* mincols, int* maxcols, int n, double tol ) nogil:
cdef inline int solve_c( double* A, double* b, double* x, int n ) nogil:

# After:
cdef inline int decompose_lu_inplace_c( double* A, int* p, int n ) noexcept nogil:
cdef inline void solve_decomposed_c( double* LU, int* p, double* b, double* x, int n ) noexcept nogil:
cdef inline void solve_decomposed_banded_c( double* LU, int* p, int* mincols, int* maxcols, double* b, double* x, int n ) noexcept nogil:
cdef inline void find_bands_c( double* LU, int* mincols, int* maxcols, int n, double tol ) noexcept nogil:
cdef inline int solve_c( double* A, double* b, double* x, int n ) noexcept nogil:
```

#### Architectural note for CC

**All `_c` functions are `cdef inline` implementations in `dgesv.pxd`, not `dgesv.pyx`.**
The `.pxd` is not just a header — it contains the full C-level implementation
(~200 lines). The `.pyx` file only contains the Python-facing wrappers that
call into these inline functions. This is intentional: the inline functions
get compiled into each downstream module that `cimport`s them, avoiding
cross-module function call overhead. **Do not move the implementations from
`.pxd` to `.pyx`.**

#### 3.6.6. No `solve_multi` — update downstream expectations

There is no `solve_multi` in this library. The batch-solving pattern is:
`lup_packed()` once, then loop calling `solve_decomposed()` (or the banded
variant). This is what pydgq's galerkin.pyx does. No changes needed here.

### 3.7. MODIFY: `pylu/dgesv.pxd`

All changes are covered in the audit table above (section 3.6.5). Summary:

1. Add `# cython: language_level=3` directive at top.
2. Replace `DEF DIAG_ELEMENT_ABS_TOL` with `cdef extern from *` / `#define` (line 24).
3. Add `noexcept` to all five `cdef inline` function signatures.
4. **Do not move, restructure, or refactor the implementations.**

### 3.8. MODERNIZE: `test/pylu_test.py` → `tests/test_dgesv.py`

Rename `test/` → `tests/` (pytest convention). Rewrite as proper pytest:

```python
"""Tests for PyLU dense linear solver."""

import numpy as np
import pytest

import pylu


class TestSolve:
    """Test pylu.solve() — combined LU + solve."""

    def test_basic_solve(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        x = pylu.solve(A, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_identity(self):
        A = np.eye(4)
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = pylu.solve(A, b)
        np.testing.assert_allclose(x, b, rtol=1e-15)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 50])
    def test_various_sizes(self, n):
        rng = np.random.default_rng(123 + n)
        A = rng.random((n, n))
        b = rng.random(n)
        x = pylu.solve(A, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_singular_raises(self):
        A = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([1.0, 1.0])
        with pytest.raises(RuntimeError):
            pylu.solve(A, b)


class TestLUP:
    """Test lup() — human-readable L, U, p decomposition."""

    def test_lup_roundtrip(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        # P A = L U, i.e. A[p, :] = L @ U
        np.testing.assert_allclose(A[p, :], L @ U, rtol=1e-12)

    def test_l_is_lower_triangular_with_unit_diag(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        np.testing.assert_allclose(np.diag(L), np.ones(5))
        assert np.allclose(L, np.tril(L))

    def test_u_is_upper_triangular(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        L, U, p = pylu.lup(A)
        assert np.allclose(U, np.triu(U))


class TestFactorizeThenSolve:
    """Test lup_packed() + solve_decomposed() workflow."""

    def test_basic(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        LU, p = pylu.lup_packed(A)
        x = pylu.solve_decomposed(LU, p, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_many_rhs_same_lhs(self):
        """The main use case: factorize once, solve with many RHS vectors."""
        rng = np.random.default_rng(42)
        n = 8
        A = rng.random((n, n))
        LU, p = pylu.lup_packed(A)

        for _ in range(20):
            b = rng.random(n)
            x = pylu.solve_decomposed(LU, p, b)
            np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_matches_one_shot_solve(self):
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        b = rng.random(5)
        x_oneshot = pylu.solve(A, b)
        LU, p = pylu.lup_packed(A)
        x_twostep = pylu.solve_decomposed(LU, p, b)
        np.testing.assert_allclose(x_oneshot, x_twostep, rtol=1e-14)


class TestBandedSolver:
    """Test find_bands() + solve_decomposed_banded() workflow."""

    def test_tridiagonal(self):
        n = 10
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < n - 1:
                A[i, i+1] = -1.0

        b = np.ones(n)
        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_banded_matches_non_banded(self):
        """Banded and non-banded solvers should give the same result."""
        rng = np.random.default_rng(42)
        n = 8
        A = rng.random((n, n))
        b = rng.random(n)

        LU, p = pylu.lup_packed(A)
        x_full = pylu.solve_decomposed(LU, p, b)

        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x_banded = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)

        np.testing.assert_allclose(x_full, x_banded, rtol=1e-14)

    def test_diagonal_matrix(self):
        n = 5
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 6.0, 12.0, 20.0, 30.0])

        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(x, [2.0, 3.0, 4.0, 5.0, 6.0], rtol=1e-14)

    def test_many_rhs_banded(self):
        """Factorize + find bands once, solve many RHS — the pydgq use case."""
        n = 6
        # Tridiagonal
        A = np.diag([2.0]*n) + np.diag([-1.0]*(n-1), k=1) + np.diag([-1.0]*(n-1), k=-1)

        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)

        rng = np.random.default_rng(42)
        for _ in range(20):
            b = rng.random(n)
            x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
            np.testing.assert_allclose(A @ x, b, rtol=1e-12)

    def test_1x1_system(self):
        A = np.array([[3.0]])
        b = np.array([9.0])
        LU, p = pylu.lup_packed(A)
        mincols, maxcols = pylu.find_bands(LU, 1e-15)
        x = pylu.solve_decomposed_banded(LU, p, mincols, maxcols, b)
        np.testing.assert_allclose(x, [3.0], rtol=1e-14)
```

**Notes:**
- Function names above match the verified public API of `dgesv.pyx`.
- The existing `pylu_test.py` likely has analytical test cases and timing code.
  Preserve the analytical tests, drop the timing benchmarks (or move to a
  separate `benchmarks/` directory).
- Add `tests/__init__.py` (empty) or configure pytest to find tests without it.

### 3.9. CREATE: `tests/test_cimport.py`

A test that verifies the `.pxd` is installed and usable:

```python
"""Test that pylu's Cython declarations are importable."""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


def test_pxd_installed():
    """Verify dgesv.pxd is present in the installed package."""
    import pylu
    pkg_dir = Path(pylu.__file__).parent
    pxd = pkg_dir / "dgesv.pxd"
    assert pxd.exists(), f"dgesv.pxd not found in {pkg_dir}"


def test_cimport_compiles():
    """Verify that 'cimport pylu.dgesv' succeeds in Cython compilation."""
    pytest.importorskip("Cython")
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx = Path(tmpdir) / "test_cimport.pyx"
        pyx.write_text(textwrap.dedent("""\
            cimport pylu.dgesv as dgesv_c
            # If this compiles, the .pxd is discoverable.
        """))
        result = subprocess.run(
            [sys.executable, "-m", "cython", str(pyx)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"cimport pylu.dgesv failed:\n{result.stderr}"
        )
```

### 3.10. CREATE: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install build dependencies
        run: pip install meson-python meson ninja Cython numpy

      - name: Install package
        run: pip install --no-build-isolation -e ".[test]"

      - name: Run tests
        run: pytest tests/ -v

  build-wheels:
    needs: test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4

      - uses: pypa/cibuildwheel@v2.21
        env:
          # Build for Python 3.10-3.13, 64-bit only
          CIBW_BUILD: "cp310-* cp311-* cp312-* cp313-*"
          CIBW_SKIP: "*-win32 *-manylinux_i686 *-musllinux*"
          CIBW_TEST_REQUIRES: pytest numpy
          CIBW_TEST_COMMAND: pytest {project}/tests/ -v

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: wheelhouse/*.whl

  sdist:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build --sdist
      - uses: actions/upload-artifact@v4
        with:
          name: sdist
          path: dist/*.tar.gz
```

**Notes:**
- `cibuildwheel` handles manylinux containers, macOS universal2, and Windows
  MSVC automatically. For a project with no OpenMP and no external C deps,
  this should Just Work™.
- `--no-build-isolation` in the test job lets us use an editable install for
  faster iteration. The wheel build job uses full isolation (cibuildwheel default).
- Skip `musllinux` and 32-bit builds — unnecessary for a scientific library.
- Python 3.14 support: add when cibuildwheel supports it (likely needs
  `CIBW_BUILD: "cp314-*"` and may need `--pre` for NumPy/Cython).
- For release publishing to PyPI, add a separate job triggered on tags.
  We can add this later.

### 3.11. OPTIONAL CREATE: `pyproject.toml` test extras

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
test = ["pytest>=7.0"]
```

---

## 4. Cython Compiler Directives

The file already has per-file directives set. Only addition needed is
`language_level=3`. The final header should be:

```
# cython: language_level=3
# cython: wraparound = False
# cython: boundscheck = False
# cython: cdivision = True
```

These can also be set in `meson.build` via `cython_args`, but the in-file
directives are preferred — they're self-documenting and visible to anyone
reading the source. Do not duplicate them in `meson.build`.

---

## 5. Migration Sequence (for CC)

Execute in this order. Each step should be a separate commit.

1. **Create `pyproject.toml` and `meson.build` files** (sections 3.2–3.4)
2. **Remove `setup.py`**
3. **Clean up `pylu/__init__.py`** — remove `__future__`, Python 2 compat
4. **Cython 3.x fixes in `dgesv.pyx` and `dgesv.pxd`** — directives, exception specs,
   remove `__future__` imports. **Do NOT rewrite working numerical code.** Only fix
   what's needed for Cython 3 compatibility. The algorithm is correct; don't touch it.
5. **Convert tests to pytest** (`tests/test_dgesv.py`)
6. **Add cimport test** (`tests/test_cimport.py`)
7. **Add CI** (`.github/workflows/ci.yml`)
8. **Update `README.md`** — new install instructions, drop Python 2 references,
   update dependency list
9. **Update `LICENSE.md`** — copyright to "2016-2026 Juha Jeronen, University of
   Jyväskylä, and JAMK University of Applied Sciences"
10. **Update `CHANGELOG.md`** — document 1.0.0 changes
11. **Tag release** — `v1.0.0`

### Verification checkpoints

After step 4: `pip install .` should succeed, `python -c "import pylu; print(pylu.solve(...))"` works.

After step 6: `pytest tests/` passes, including the cimport test.

After step 7: CI is green on all three platforms.

---

## 6. Things CC Must NOT Do

- **Do not refactor the numerical algorithm.** The LU decomposition code is
  mathematically correct and performance-tested. Modernize the build system
  and fix Cython 3 compatibility; leave the numerics alone.
- **Do not switch from raw pointers to memoryviews in the `_c` (nogil) functions.**
  The raw pointer interface is the whole point — it's what downstream `cimport`
  users call from their own `nogil` blocks. The Python-facing wrappers can
  optionally use memoryviews, but the `_c` functions must keep their signatures.
- **Do not add type annotations to `.pyx` files** beyond Cython's own type
  declarations. Cython has its own type system; Python type hints in `.pyx`
  can cause confusion or conflicts.
- **Do not vendor NumPy headers.** Use the `dependency('numpy')` mechanism
  in meson or the `numpy.get_include()` fallback.
- **Do not add Python 3.14 t-string support** or any other forward-looking
  feature. This is a C-level numerics library; it doesn't use any string
  formatting features that would be affected.

---

## 7. Platform-Specific Notes

### Linux
No special handling needed. GCC is the default compiler.

### macOS
- meson auto-detects `clang`. No OpenMP needed for PyLU, so no `libomp` issues.
- For ARM (Apple Silicon) + x86_64 universal2 builds, cibuildwheel handles this.
- Compiler flags: avoid `-march=native` in wheels (not portable). meson's
  `buildtype=release` gives `-O2` which is fine. Users building from source
  can add `-march=native` via `CFLAGS`.

### Windows
- meson auto-detects MSVC. Cython compiles to C, MSVC compiles the C.
- No special handling expected for a project this simple.
- Potential gotcha: MSVC doesn't support C99 `restrict` keyword. Cython
  generates `__restrict` for MSVC automatically, but verify.

---

## 8. Resolved Decisions

1. **Email in pyproject.toml:** `juha.jeronen@jamk.fi`
2. **Package name on PyPI:** Keep `pylu`. Existing PyPI entry is ours.
3. **License copyright:** Update to "2016-2026 Juha Jeronen, University of
   Jyväskylä, and JAMK University of Applied Sciences."
4. **Banded solver:** Part of the public API. Tests must cover it.
5. **API inventory (verified from source):** There is no `solve_multi` — that was
   a misremembering. The actual public Python API is:
   - `solve(A, b)` — one-shot LU + solve
   - `lup(A)` → `(L, U, p)` — human-readable LU decomposition
   - `lup_packed(A)` → `(LU, p)` — packed LU for factorize-once workflows
   - `solve_decomposed(LU, p, b)` — solve with pre-factored LU (same LHS, many RHS)
   - `find_bands(LU, tol)` → `(mincols, maxcols)` — detect band structure
   - `solve_decomposed_banded(LU, p, mincols, maxcols, b)` — banded solve
   The C-level nogil API (`_c` suffix): `solve_c`, `decompose_lu_inplace_c`,
   `solve_decomposed_c`, `find_bands_c`, `solve_decomposed_banded_c`.
   Tests must cover all six Python functions and the factorize-once + banded workflows.
6. **Version:** 1.0.0.
