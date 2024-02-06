from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *
from blaslib import *
from codegen_helpers import *
from parameters import Level_1_Params, Level_2_Params


### EXO_LOC ALGORITHM START ###
@proc
def trmv_rm_un_template(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        xCopy[n - i - 1] = 0.0
        for j in seq(0, i):
            xCopy[n - i - 1] += A[n - i - 1, n - j - 1] * x[n - j - 1]
        if Diag == 0:
            xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
        else:
            xCopy[n - i - 1] += x[n - i - 1]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_ln_template(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        xCopy[i] = 0.0
        for j in seq(0, i):
            xCopy[i] += A[i, j] * x[j]
        if Diag == 0:
            xCopy[i] += A[i, i] * x[i]
        else:
            xCopy[i] += x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_ut_template(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for l in seq(0, n):
        xCopy[l] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]
        if Diag == 0:
            xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
        else:
            xCopy[n - i - 1] += x[n - i - 1]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_lt_template(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for l in seq(0, n):
        xCopy[l] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]
        if Diag == 0:
            xCopy[i] += A[i, i] * x[i]
        else:
            xCopy[i] += x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###

template_sched_list = [
    trmv_rm_un_template,
    trmv_rm_ln_template,
    trmv_rm_ut_template,
    trmv_rm_lt_template,
]

for precision in ("f32", "f64"):
    for template in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = generate_stride_1_proc(template, precision)
        level_2_params = Level_2_Params(
            precision=precision,
            rows_interleave_factor=4,
            interleave_factor=2,
            accumulators_count=2,
        )
        proc_stride_1 = optimize_level_2(
            proc_stride_1,
            proc_stride_1.find_loop("i"),
            level_2_params,
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
