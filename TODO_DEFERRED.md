## Deferred issues

- **Memory leak in `solve_c` on failure path** (`pylu/dgesv.pxd`, ~line 229): If `decompose_lu_inplace_c` returns False, `solve_c` returns without freeing `p` and `LU`. Pre-existing bug, not introduced by migration.
