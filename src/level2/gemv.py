from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def gemv_rm_nt(m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][n], y: [R][m]):
    assert stride(A, 1) == 1

    for i in seq(0, m):
        y[i] = y[i] * beta
        for j in seq(0, n):
            y[i] += alpha * (x[j] * A[i, j])


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###

gemv_rm_t = gemv_rm_nt.transpose(gemv_rm_nt.args()[4])
gemv_rm_t = rename(gemv_rm_t, "gemv_rm_t")

gemv_rm_nt = stage_mem(
    gemv_rm_nt, gemv_rm_nt.find_loop("j"), "y[i]", "result", accum=True
)
gemv_rm_nt = lift_reduce_constant(gemv_rm_nt, gemv_rm_nt.find_loop("j").expand(1, 0))

gemv_rm_t = fission(gemv_rm_t, gemv_rm_t.find_loop("j").before())
gemv_rm_t = reorder_loops(gemv_rm_t, gemv_rm_t.find_loop("i #1"))
gemv_rm_t = left_reassociate_expr(gemv_rm_t, gemv_rm_t.find("alpha * _"))

variants_generator(optimize_level_2)(gemv_rm_nt, "i", 4, 2, globals=globals())
variants_generator(optimize_level_2)(gemv_rm_t, "j", 4, 2, globals=globals())

### EXO_LOC SCHEDULE END ###
