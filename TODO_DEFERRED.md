# Deferred TODOs

## Sanity-check numerics on the first MSVC-built Windows wheels

Windows wheels up to and including the current release were built with MinGW-w64 gcc — meson
picked it up off the runner PATH, and nothing flagged it because the extensions link no
OpenMP and so imported cleanly. CI now activates MSVC, so the *next* release ships wheels
from a different compiler than every previous one.

The C API side is fine; the part worth a look is floating point, where gcc and MSVC differ
in default contraction and optimization of expressions. For an LU solver that is a plausible
place for last-bit differences to show up. Before publishing, run the suite on a Windows
wheel and compare residuals against the Linux build rather than assuming bit-identical
output.

Discovered while adding MSVC activation to CI (2026-08-14).

## Benchmark cross-module `cdef` call overhead vs. `cdef inline` in `.pxd`

The inline-in-pxd architecture was chosen to avoid cross-module function call overhead. This was measured under Python 2.7 / Cython 0.x. Modern Cython 3.x and CPython 3.11+ may have narrowed the gap — worth re-measuring to see if the architecture is still justified, or if a simpler `.pyx`-only layout would perform comparably.
