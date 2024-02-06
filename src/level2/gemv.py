from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API import compile_procs

from blas_common_schedules import *
import exo_blas_config as C
from composed_schedules import *
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


gemv_rm_t = gemv_rm_nt.transpose(gemv_rm_nt.args()[4])
gemv_rm_t = rename(gemv_rm_t, "gemv_rm_t")

gemv_rm_nt = stage_mem(
    gemv_rm_nt, gemv_rm_nt.find_loop("j"), "y[i]", "result", accum=True
)
gemv_rm_nt = lift_reduce_constant(gemv_rm_nt, gemv_rm_nt.find_loop("j").expand(1, 0))

gemv_rm_t = fission(gemv_rm_t, gemv_rm_t.find_loop("j").before())
gemv_rm_t = reorder_loops(gemv_rm_t, gemv_rm_t.find_loop("i #1"))
gemv_rm_t = left_reassociate_expr(gemv_rm_t, gemv_rm_t.find("alpha * _"))

### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###

template_sched_list = [
    (gemv_rm_nt, "i"),
    (gemv_rm_t, "j"),
]

for precision in ("f32", "f64"):
    for template, it in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = generate_stride_1_proc(template, precision)
        proc_stride_1 = optimize_level_2(
            proc_stride_1, proc_stride_1.find_loop(it), precision, C.Machine, 4, 2
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
