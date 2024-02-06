from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


@proc
def tbmv_row_major_Upper_NonTrans_template(
    n: size, k: size, x: [R][n], A: [R][n, k + 1], Diag: size
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    for i in seq(0, n - k):
        dot: R
        dot = 0.0

        if Diag == 0:
            dot = x[i] * A[i, 0]
        else:
            dot = x[i]

        for j in seq(0, k):
            dot += A[i, j + 1] * x[i + j + 1]

        x[i] = dot

    for i in seq(0, k):
        dot: R
        dot = 0.0

        if Diag == 0:
            dot = x[n - k + i] * A[n - k + i, 0]
        else:
            dot = x[n - k + i]

        for j in seq(0, k - i - 1):
            dot += A[n - k + i, j + 1] * x[n - k + i + j + 1]

        x[n - k + i] = dot


@proc
def tbmv_row_major_Lower_NonTrans_template(
    n: size, k: size, x: [R][n], A: [R][n, k + 1], Diag: size
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    for i in seq(0, n - k):
        dot: R
        dot = 0.0

        for j in seq(0, k):
            dot += A[n - i - 1, k - j - 1] * x[n - i - 1 - j - 1]

        if Diag == 0:
            dot += x[n - i - 1] * A[n - i - 1, k]
        else:
            dot += x[n - i - 1]

        x[n - i - 1] = dot

    for i in seq(0, k):
        dot: R
        dot = 0.0

        for j in seq(0, k - i - 1):
            dot += A[n - (n - k + i) - 1, k - j - 1] * x[n - (n - k + i) - 1 - j - 1]

        if Diag == 0:
            dot += x[n - (n - k + i) - 1] * A[n - (n - k + i) - 1, k]
        else:
            dot += x[n - (n - k + i) - 1]

        x[n - (n - k + i) - 1] = dot


@proc
def tbmv_row_major_Upper_Trans_template(
    n: size, k: size, x: [R][n], A: [R][n, k + 1], Diag: size
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    xRes: R[n]
    for i in seq(0, n):
        xRes[i] = 0.0

    for i in seq(0, n - k):
        if Diag == 0:
            xRes[i] += x[i] * A[i, 0]
        else:
            xRes[i] += x[i]

        for j in seq(0, k):
            xRes[i + j + 1] += A[i, j + 1] * x[i]

    for i in seq(0, k):
        if Diag == 0:
            xRes[n - k + i] += x[n - k + i] * A[n - k + i, 0]
        else:
            xRes[n - k + i] += x[n - k + i]

        for j in seq(0, k - i - 1):
            xRes[n - k + i + j + 1] += A[n - k + i, j + 1] * x[n - k + i]

    for i in seq(0, n):
        x[i] = xRes[i]


@proc
def tbmv_row_major_Lower_Trans_template(
    n: size, k: size, x: [R][n], A: [R][n, k + 1], Diag: size
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    xRes: R[n]
    for i in seq(0, n):
        xRes[i] = 0.0

    for i in seq(0, n - k):

        for j in seq(0, k):
            xRes[n - i - 1 - j - 1] += A[n - i - 1, k - j - 1] * x[n - i - 1]

        if Diag == 0:
            xRes[n - i - 1] += x[n - i - 1] * A[n - i - 1, k]
        else:
            xRes[n - i - 1] += x[n - i - 1]

    for i in seq(0, k):

        for j in seq(0, k - i - 1):
            xRes[n - (n - k + i) - 1 - j - 1] += (
                A[n - (n - k + i) - 1, k - j - 1] * x[n - (n - k + i) - 1]
            )

        if Diag == 0:
            xRes[n - (n - k + i) - 1] += (
                x[n - (n - k + i) - 1] * A[n - (n - k + i) - 1, k]
            )
        else:
            xRes[n - (n - k + i) - 1] += x[n - (n - k + i) - 1]

    for i in seq(0, n):
        x[i] = xRes[i]


def specialize_tbmv(tbmv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = tbmv.name()
    name = name.replace("_template", "")
    specialized = rename(tbmv, "exo_" + prefix + name)

    args = ["x", "A"]

    if "_Trans_" in tbmv.name():
        args.append("xRes")
    else:
        args.append("dot")
        args.append("dot #1")

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_interleave_tbmv_row_major_stride_1(
    tbmv, VEC_W, ROW_INTERLEAVE_FACTOR, memory, instructions, precision
):
    stride_1 = specialize_tbmv(tbmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    return stride_1


#################################################
# Kernel Parameters
#################################################

ROW_INTERLEAVE_FACTOR = C.Machine.vec_units

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_stbmv_row_major_Upper_NonTrans_stride_any = specialize_tbmv(
    tbmv_row_major_Upper_NonTrans_template, "f32"
)
exo_stbmv_row_major_Upper_NonTrans_stride_any = rename(
    exo_stbmv_row_major_Upper_NonTrans_stride_any,
    exo_stbmv_row_major_Upper_NonTrans_stride_any.name() + "_stride_any",
)
exo_stbmv_row_major_Lower_NonTrans_stride_any = specialize_tbmv(
    tbmv_row_major_Lower_NonTrans_template, "f32"
)
exo_stbmv_row_major_Lower_NonTrans_stride_any = rename(
    exo_stbmv_row_major_Lower_NonTrans_stride_any,
    exo_stbmv_row_major_Lower_NonTrans_stride_any.name() + "_stride_any",
)
exo_stbmv_row_major_Upper_Trans_stride_any = specialize_tbmv(
    tbmv_row_major_Upper_Trans_template, "f32"
)
exo_stbmv_row_major_Upper_Trans_stride_any = rename(
    exo_stbmv_row_major_Upper_Trans_stride_any,
    exo_stbmv_row_major_Upper_Trans_stride_any.name() + "_stride_any",
)
exo_stbmv_row_major_Lower_Trans_stride_any = specialize_tbmv(
    tbmv_row_major_Lower_Trans_template, "f32"
)
exo_stbmv_row_major_Lower_Trans_stride_any = rename(
    exo_stbmv_row_major_Lower_Trans_stride_any,
    exo_stbmv_row_major_Lower_Trans_stride_any.name() + "_stride_any",
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_reduce_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_stbmv_row_major_Upper_NonTrans_stride_1 = (
    schedule_interleave_tbmv_row_major_stride_1(
        tbmv_row_major_Upper_NonTrans_template,
        C.Machine.f32_vec_width,
        ROW_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_stbmv_row_major_Lower_NonTrans_stride_1 = (
    schedule_interleave_tbmv_row_major_stride_1(
        tbmv_row_major_Lower_NonTrans_template,
        C.Machine.f32_vec_width,
        ROW_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_stbmv_row_major_Upper_Trans_stride_1 = schedule_interleave_tbmv_row_major_stride_1(
    tbmv_row_major_Upper_Trans_template,
    C.Machine.f32_vec_width,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_stbmv_row_major_Lower_Trans_stride_1 = schedule_interleave_tbmv_row_major_stride_1(
    tbmv_row_major_Lower_Trans_template,
    C.Machine.f32_vec_width,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dtbmv_row_major_Upper_NonTrans_stride_any = specialize_tbmv(
    tbmv_row_major_Upper_NonTrans_template, "f64"
)
exo_dtbmv_row_major_Upper_NonTrans_stride_any = rename(
    exo_dtbmv_row_major_Upper_NonTrans_stride_any,
    exo_dtbmv_row_major_Upper_NonTrans_stride_any.name() + "_stride_any",
)
exo_dtbmv_row_major_Lower_NonTrans_stride_any = specialize_tbmv(
    tbmv_row_major_Lower_NonTrans_template, "f64"
)
exo_dtbmv_row_major_Lower_NonTrans_stride_any = rename(
    exo_dtbmv_row_major_Lower_NonTrans_stride_any,
    exo_dtbmv_row_major_Lower_NonTrans_stride_any.name() + "_stride_any",
)
exo_dtbmv_row_major_Upper_Trans_stride_any = specialize_tbmv(
    tbmv_row_major_Upper_Trans_template, "f64"
)
exo_dtbmv_row_major_Upper_Trans_stride_any = rename(
    exo_dtbmv_row_major_Upper_Trans_stride_any,
    exo_dtbmv_row_major_Upper_Trans_stride_any.name() + "_stride_any",
)
exo_dtbmv_row_major_Lower_Trans_stride_any = specialize_tbmv(
    tbmv_row_major_Lower_Trans_template, "f64"
)
exo_dtbmv_row_major_Lower_Trans_stride_any = rename(
    exo_dtbmv_row_major_Lower_Trans_stride_any,
    exo_dtbmv_row_major_Lower_Trans_stride_any.name() + "_stride_any",
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.fmadd_reduce_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dtbmv_row_major_Upper_NonTrans_stride_1 = (
    schedule_interleave_tbmv_row_major_stride_1(
        tbmv_row_major_Upper_NonTrans_template,
        C.Machine.f32_vec_width // 2,
        ROW_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtbmv_row_major_Lower_NonTrans_stride_1 = (
    schedule_interleave_tbmv_row_major_stride_1(
        tbmv_row_major_Lower_NonTrans_template,
        C.Machine.f32_vec_width // 2,
        ROW_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtbmv_row_major_Upper_Trans_stride_1 = schedule_interleave_tbmv_row_major_stride_1(
    tbmv_row_major_Upper_Trans_template,
    C.Machine.f32_vec_width // 2,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dtbmv_row_major_Lower_Trans_stride_1 = schedule_interleave_tbmv_row_major_stride_1(
    tbmv_row_major_Lower_Trans_template,
    C.Machine.f32_vec_width // 2,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)

entry_points = [
    exo_stbmv_row_major_Upper_NonTrans_stride_any,
    exo_stbmv_row_major_Upper_NonTrans_stride_1,
    exo_dtbmv_row_major_Upper_NonTrans_stride_any,
    exo_dtbmv_row_major_Upper_NonTrans_stride_1,
    exo_stbmv_row_major_Lower_NonTrans_stride_any,
    exo_stbmv_row_major_Lower_NonTrans_stride_1,
    exo_dtbmv_row_major_Lower_NonTrans_stride_any,
    exo_dtbmv_row_major_Lower_NonTrans_stride_1,
    exo_stbmv_row_major_Upper_Trans_stride_any,
    exo_stbmv_row_major_Upper_Trans_stride_1,
    exo_dtbmv_row_major_Upper_Trans_stride_any,
    exo_dtbmv_row_major_Upper_Trans_stride_1,
    exo_stbmv_row_major_Lower_Trans_stride_any,
    exo_stbmv_row_major_Lower_Trans_stride_1,
    exo_dtbmv_row_major_Lower_Trans_stride_any,
    exo_dtbmv_row_major_Lower_Trans_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
