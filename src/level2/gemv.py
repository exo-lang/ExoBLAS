from __future__ import annotations
import os
import sys
from pathlib import Path

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
from parameters import Level_1_Params, Level_2_Params

### EXO_LOC ALGORITHM START ###
@proc
def gemv_row_major_NonTrans(
    m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][n], y: [R][m]
):
    assert stride(A, 1) == 1

    for i in seq(0, m):
        result: R
        result = 0.0
        for j in seq(0, n):
            result += x[j] * A[i, j]
        y[i] = beta * y[i] + alpha * result


@proc
def gemv_row_major_Trans(
    m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][m], y: [R][n]
):
    assert stride(A, 1) == 1

    for k in seq(0, n):
        y[k] = beta * y[k]
    for i in seq(0, m):
        alphaXi: R
        alphaXi = alpha * x[i]
        for j in seq(0, n):
            y[j] += alphaXi * A[i, j]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###

template_sched_list = [
    (optimize_level_2, gemv_row_major_NonTrans, "x[j]"),
    (optimize_level_2, gemv_row_major_Trans, "y[j]"),
]

for precision in ("f32", "f64"):
    for sched, template, reuse in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        params = Level_2_Params(
            precision=precision,
            rows_interleave_factor=4,
        )
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = generate_stride_1_proc(template, precision)
        proc_stride_1 = sched(
            proc_stride_1,
            proc_stride_1.find_loop("i"),
            params,
            reuse,
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
