## Deferred issues

- **Single source of truth for version**: Currently the version is in `__init__.py`, `pyproject.toml`, and `meson.build`. Consolidate to one place. meson-python supports `dynamic = ["version"]` in pyproject.toml — investigate how to chain this so `__init__.py` is the sole source.
- **Benchmark cross-module `cdef` call overhead vs. `cdef inline` in `.pxd`**: The inline-in-pxd architecture was chosen to avoid cross-module function call overhead. This was measured under Python 2.7 / Cython 0.x. Modern Cython 3.x and CPython 3.11+ may have narrowed the gap — worth re-measuring to see if the architecture is still justified, or if a simpler `.pyx`-only layout would perform comparably.
