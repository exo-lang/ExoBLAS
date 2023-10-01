from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    vectorize_to_loops,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params, Level_2_Params


### EXO_LOC ALGORITHM START ###
@proc
def trmv_row_major_Upper_NonTrans_Unit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
        xCopy[n - i - 1] = dot

    for l in seq(0, n):
        x[l] += xCopy[l]


@proc
def trmv_row_major_Upper_NonTrans_NonUnit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
        xCopy[n - i - 1] = dot + A[n - i - 1, n - i - 1] * x[n - i - 1]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_row_major_Lower_NonTrans_Unit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[i, j] * x[j]
        xCopy[i] = dot

    for l in seq(0, n):
        x[l] += xCopy[l]


@proc
def trmv_row_major_Lower_NonTrans_NonUnit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[i, j] * x[j]
        xCopy[i] = dot + A[i, i] * x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_row_major_Upper_Trans_Unit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]

    for i in seq(0, n):
        x[i] += xCopy[i]


@proc
def trmv_row_major_Upper_Trans_NonUnit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]
        xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]

    for i in seq(0, n):
        x[i] = xCopy[i]


@proc
def trmv_row_major_Lower_Trans_Unit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]

    for i in seq(0, n):
        x[i] += xCopy[i]


@proc
def trmv_row_major_Lower_Trans_NonUnit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]
        xCopy[i] += A[i, i] * x[i]

    for i in seq(0, n):
        x[i] = xCopy[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_trmv_row_major_vectorize_reuse_over_rows(
    trmv, level_2_params, level_1_params
):
    isNonTrans = "NonTrans" in trmv.name()

    trmv = generate_stride_1_proc(trmv, level_1_params.precision)
    level_2_params.instructions = None
    inner_loop = trmv.find_loop("j")
    outer_loop = inner_loop.parent()
    return trmv
    trmv = blas_vectorize(trmv, inner_loop, level_2_params)
    trmv = simplify(trmv)
    trmv = interleave_outer_loop_with_inner_loop(
        trmv,
        outer_loop,
        inner_loop,
        min(level_2_params.rows_interleave_factor, level_2_params.vec_width),
    )
    trmv = replace_all(trmv, C.Machine.get_instructions(level_2_params.precision))
    trmv = unroll_loop(trmv, trmv.find_loop("ii"))
    trmv = apply_to_block(trmv, trmv.find_loop("ii").body(), hoist_stmt)
    trmv = unroll_loop(trmv, trmv.find_loop("ii"))
    if isNonTrans:
        dot_alloc = trmv.find("dot : _")
        trmv = set_memory(trmv, "dot", DRAM_STATIC)
        trmv = unroll_loop(trmv, trmv.find_loop("ii"))
    trmv = simplify(trmv)

    return trmv


template_sched_list = [
    trmv_row_major_Lower_NonTrans_NonUnit_template,
    trmv_row_major_Upper_NonTrans_NonUnit_template,
    trmv_row_major_Lower_NonTrans_Unit_template,
    trmv_row_major_Upper_NonTrans_Unit_template,
    trmv_row_major_Lower_Trans_NonUnit_template,
    trmv_row_major_Upper_Trans_NonUnit_template,
    trmv_row_major_Lower_Trans_Unit_template,
    trmv_row_major_Upper_Trans_Unit_template,
]

for precision in ("f32", "f64"):
    for template in template_sched_list:
        if "NonTrans" in template.name():
            level_2_params = Level_2_Params(
                precision=precision,
                rows_interleave_factor=8,
                interleave_factor=2,
                accumulators_count=1,
            )
        else:
            level_2_params = Level_2_Params(
                precision=precision,
                rows_interleave_factor=4,
                interleave_factor=4,
                accumulators_count=1,
            )
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = schedule_trmv_row_major_vectorize_reuse_over_rows(
            template,
            level_2_params,
            Level_1_Params(precision=precision),
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
