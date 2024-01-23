from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


@proc
def sbmv_scal_y(n: size, beta: R, y: [R][n]):

    for i in seq(0, n):
        y[i] = beta * y[i]


@proc
def sbmv_row_major_Upper_template(
    n: size, k: size, alpha: R, A: [R][n, k + 1], x: [R][n], y: [R][n]
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    for i in seq(0, n):
        temp: R
        temp = alpha * x[i]
        dot: R
        dot = 0.0
        for j in seq(0, k):
            if i + j + 1 < n:
                y[i + j + 1] += temp * A[i, j + 1]
                dot += A[i, j + 1] * x[i + j + 1]
        y[i] += temp * A[i, 0] + alpha * dot


@proc
def sbmv_row_major_Lower_template(
    n: size, k: size, alpha: R, A: [R][n, k + 1], x: [R][n], y: [R][n]
):
    assert stride(A, 1) == 1
    assert k <= n - 1

    for i in seq(0, n):
        temp: R
        temp = alpha * x[i]
        dot: R
        dot = 0.0
        for j in seq(0, k):
            if i - j - 1 >= 0:
                y[i - j - 1] += temp * A[i, k - j - 1]
                dot += A[i, k - j - 1] * x[i - j - 1]
        y[i] += temp * A[i, k] + alpha * dot


def specialize_sbmv(sbmv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = sbmv.name()
    name = name.replace("_template", "")
    specialized = rename(sbmv, "exo_" + prefix + name)

    if "scal" in sbmv.name():
        args = ["y", "beta"]
    else:
        args = ["x", "A", "alpha", "y", "temp", "dot"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_interleave_sbmv_scal_y(
    VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    stride_1 = specialize_sbmv(sbmv_scal_y, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    return stride_1


def schedule_interleave_sbmv_row_major_stride_1(
    sbmv, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    stride_1 = specialize_sbmv(sbmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    return stride_1


#################################################
# Kernel Parameters
#################################################

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_ssbmv_scal_y_stride_any = specialize_sbmv(sbmv_scal_y, "f32")
exo_ssbmv_scal_y_stride_any = rename(
    exo_ssbmv_scal_y_stride_any, exo_ssbmv_scal_y_stride_any.name() + "_stride_any"
)
exo_ssbmv_row_major_Upper_stride_any = specialize_sbmv(
    sbmv_row_major_Upper_template, "f32"
)
exo_ssbmv_row_major_Upper_stride_any = rename(
    exo_ssbmv_row_major_Upper_stride_any,
    exo_ssbmv_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_ssbmv_row_major_Lower_stride_any = specialize_sbmv(
    sbmv_row_major_Lower_template, "f32"
)
exo_ssbmv_row_major_Lower_stride_any = rename(
    exo_ssbmv_row_major_Lower_stride_any,
    exo_ssbmv_row_major_Lower_stride_any.name() + "_stride_any",
)
f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_reduce_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_ssbmv_row_major_Upper_stride_1 = schedule_interleave_sbmv_row_major_stride_1(
    sbmv_row_major_Upper_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_ssbmv_row_major_Lower_stride_1 = schedule_interleave_sbmv_row_major_stride_1(
    sbmv_row_major_Lower_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_ssbmv_scal_y_stride_1 = schedule_interleave_sbmv_scal_y(
    C.Machine.vec_width, 1, C.Machine.mem_type, f32_instructions, "f32"
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dsbmv_scal_y_stride_any = specialize_sbmv(sbmv_scal_y, "f64")
exo_dsbmv_scal_y_stride_any = rename(
    exo_dsbmv_scal_y_stride_any, exo_dsbmv_scal_y_stride_any.name() + "_stride_any"
)
exo_dsbmv_row_major_Upper_stride_any = specialize_sbmv(
    sbmv_row_major_Upper_template, "f64"
)
exo_dsbmv_row_major_Upper_stride_any = rename(
    exo_dsbmv_row_major_Upper_stride_any,
    exo_dsbmv_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_dsbmv_row_major_Lower_stride_any = specialize_sbmv(
    sbmv_row_major_Lower_template, "f64"
)
exo_dsbmv_row_major_Lower_stride_any = rename(
    exo_dsbmv_row_major_Lower_stride_any,
    exo_dsbmv_row_major_Lower_stride_any.name() + "_stride_any",
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.fmadd_reduce_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dsbmv_row_major_Upper_stride_1 = schedule_interleave_sbmv_row_major_stride_1(
    sbmv_row_major_Upper_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dsbmv_row_major_Lower_stride_1 = schedule_interleave_sbmv_row_major_stride_1(
    sbmv_row_major_Lower_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dsbmv_scal_y_stride_1 = schedule_interleave_sbmv_scal_y(
    C.Machine.vec_width // 2, 1, C.Machine.mem_type, f64_instructions, "f64"
)

entry_points = [
    exo_ssbmv_scal_y_stride_any,
    exo_ssbmv_scal_y_stride_1,
    exo_ssbmv_row_major_Upper_stride_any,
    exo_ssbmv_row_major_Upper_stride_1,
    exo_dsbmv_row_major_Upper_stride_any,
    exo_dsbmv_row_major_Upper_stride_1,
    exo_ssbmv_row_major_Lower_stride_any,
    exo_ssbmv_row_major_Lower_stride_1,
    exo_dsbmv_row_major_Lower_stride_any,
    exo_dsbmv_row_major_Lower_stride_1,
    exo_dsbmv_scal_y_stride_any,
    exo_dsbmv_scal_y_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
