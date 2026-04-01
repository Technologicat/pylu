# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is PyLU

Small Cython library providing LU decomposition with partial pivoting for solving `A x = b`. BSD-licensed. The raison d'être is calling the solver inside `nogil` blocks without depending on SciPy. Exposes both Python and Cython (`cimport`) APIs.

Single runtime dependency: NumPy. This constraint is intentional — keep it that way.

## Build and Development

Uses meson-python as build backend, PDM for dependency management. Python ≥ 3.11.

```bash
pdm config venv.in_project true
pdm use 3.14
pdm install
```

**Editable installs:** meson-python editable installs rebuild the extension on import. After modifying `.pyx` or `.pxd` files, re-run `pdm install` to rebuild. Alternatively, use a non-editable install (`pip install .`) and reinstall after changes.

## Running Tests

```bash
pdm run pytest tests/ -v
```

22 tests covering all 6 public functions, edge cases (1×1, identity, singular), the multi-RHS workflow, banded vs. non-banded equivalence, and cimport verification.

## Architecture

### The .pxd contains the implementations

This is the key architectural decision. `dgesv.pxd` is not just a header — it contains ~200 lines of `cdef inline` C-level implementations. `dgesv.pyx` only contains thin Python-facing wrappers that convert memoryviews to raw pointers and call the `_c` functions in `nogil` blocks.

This is intentional: the inline functions get compiled into each downstream module that `cimport`s them, avoiding cross-module function call overhead. **Do not move implementations from `.pxd` to `.pyx`.**

### Public Python API (dgesv.pyx)

- `solve(A, b)` — one-shot LU + solve
- `lup(A)` → `(L, U, p)` — human-readable decomposition
- `lup_packed(A)` → `(LU, p)` — packed LU for factorize-once workflows
- `solve_decomposed(LU, p, b)` — solve with pre-factored LU
- `find_bands(LU, tol)` → `(mincols, maxcols)` — detect band structure in LU
- `solve_decomposed_banded(LU, p, mincols, maxcols, b)` — banded solve

### Cython nogil API (dgesv.pxd)

All `cdef inline ... noexcept nogil`:

- `decompose_lu_inplace_c` — LU factorization with partial pivoting
- `solve_decomposed_c` — forward/backward substitution
- `solve_decomposed_banded_c` — band-aware substitution
- `find_bands_c` — band structure detection
- `solve_c` — combined factorize + solve (allocates temp memory)

### Critical constraint: .pxd installation

The `.pxd` file **must** be installed alongside the compiled `.so` so that downstream projects (pydgq, python-wlsqm) can `cimport pylu.dgesv`. This is handled by `py.install_sources('dgesv.pxd', ...)` in `pylu/meson.build`. Always verify after build changes:

```bash
pdm run python -c "import pylu; from pathlib import Path; print(list(Path(pylu.__file__).parent.glob('*.pxd')))"
```

## Key Rules

- **Do not refactor the numerical algorithm.** The LU code is mathematically correct and performance-tested. Only fix compatibility issues.
- **Do not switch from raw pointers to memoryviews in the `_c` functions.** The raw pointer interface is the whole point — downstream `cimport` users call these from their own `nogil` blocks.
- **Do not add Python type annotations to `.pyx` files.** Cython has its own type system.
- **Do not hardcode `-march=native` or similar arch-specific flags.** Meson's `buildtype=release` gives `-O2`. Users building from source can set `CFLAGS` if desired.
