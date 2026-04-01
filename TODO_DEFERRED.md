## Deferred issues

- **Benchmark cross-module `cdef` call overhead vs. `cdef inline` in `.pxd`**: The inline-in-pxd architecture was chosen to avoid cross-module function call overhead. This was measured under Python 2.7 / Cython 0.x. Modern Cython 3.x and CPython 3.11+ may have narrowed the gap — worth re-measuring to see if the architecture is still justified, or if a simpler `.pyx`-only layout would perform comparably.
