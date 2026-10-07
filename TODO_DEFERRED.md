# Deferred TODOs

<!-- New items go below this line. -->

## Benchmark cross-module `cdef` call overhead vs. `cdef inline` in `.pxd`

*Cluster: performance · Cost: M · Gate: none · Filed: 2026-04-17*

The inline-in-pxd architecture was chosen to avoid cross-module function call overhead. This was measured under Python 2.7 / Cython 0.x. Modern Cython 3.x and CPython 3.11+ may have narrowed the gap — worth re-measuring to see if the architecture is still justified, or if a simpler `.pyx`-only layout would perform comparably.
