from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC, DRAM
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
)
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params, Level_2_Params

class PACKED_1D(DRAM):
    @classmethod
    def custom_read(cls, baseptr, indices, srcinfo):
        assert len(indices) == 2
        i_expr = f"{indices[0]} * ({indices[0]} + 1) / 2 "
        j_expr = indices[1]
        offset = i_expr + " + " + j_expr

        return f"{baseptr}[{offset}]"


### EXO_LOC ALGORITHM START ###
@proc
def tpmv_row_major_Upper_NonTrans_Unit_template(n: size, x: [R][n], A: R[n, n] @ PACKED_1D):
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
def tpmv_row_major_Upper_NonTrans_NonUnit_template(n: size, x: [R][n], A: R[n, n] @ PACKED_1D):
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
def tpmv_row_major_Lower_NonTrans_Unit_template(n: size, x: [R][n], A: R[n, n] @ PACKED_1D):
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
def tpmv_row_major_Lower_NonTrans_NonUnit_template(n: size, x: [R][n], A: R[n, n] @ PACKED_1D):
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
def tpmv_row_major_Upper_Trans_Unit_template(
    n: size, x: [R][n], A: R[n, n] @ PACKED_1D, Diag: size
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
def tpmv_row_major_Upper_Trans_NonUnit_template(
    n: size, x: [R][n], A: R[n, n] @ PACKED_1D, Diag: size
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
def tpmv_row_major_Lower_Trans_Unit_template(
    n: size, x: [R][n], A: R[n, n] @ PACKED_1D, Diag: size
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
def tpmv_row_major_Lower_Trans_NonUnit_template(
    n: size, x: [R][n], A: R[n, n] @ PACKED_1D, Diag: size
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

template_sched_list = [
    tpmv_row_major_Lower_NonTrans_NonUnit_template,
    tpmv_row_major_Upper_NonTrans_NonUnit_template,
    tpmv_row_major_Lower_NonTrans_Unit_template,
    tpmv_row_major_Upper_NonTrans_Unit_template,
    tpmv_row_major_Lower_Trans_NonUnit_template,
    tpmv_row_major_Upper_Trans_NonUnit_template,
    tpmv_row_major_Lower_Trans_Unit_template,
    tpmv_row_major_Upper_Trans_Unit_template,
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

### EXO_LOC SCHEDULE END ###
