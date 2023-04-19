from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C

from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)


@proc
def syr_row_major_Upper_template(n: size, alpha: R, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, n - i):
            A[i, i + j] += alpha * x[i] * x[i + j]


@proc
def syr_row_major_Lower_template(n: size, alpha: R, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, i + 1):
            A[i, j] += alpha * x[i] * x[j]


def specialize_syr(syr, precision):
    prefix = "s" if precision == "f32" else "d"
    name = syr.name()
    name = name.replace("_template", "")
    specialized = rename(syr, "exo_" + prefix + name)

    args = ["alpha", "x", "A"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_interleave_syr_row_major_stride_1(
    syr, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    stride_1 = specialize_syr(syr, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    j_loop = stride_1.find_loop("j")
    stride_1 = vectorize(stride_1, j_loop, VEC_W, memory, precision)
    stride_1 = apply_to_block(stride_1, stride_1.forward(j_loop).body(), hoist_stmt)
    stride_1 = replace_all(stride_1, instructions)

    return stride_1


#################################################
# Kernel Parameters
#################################################

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_ssyr_row_major_Upper_stride_any = specialize_syr(
    syr_row_major_Upper_template, "f32"
)
exo_ssyr_row_major_Upper_stride_any = rename(
    exo_ssyr_row_major_Upper_stride_any,
    exo_ssyr_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_ssyr_row_major_Lower_stride_any = specialize_syr(
    syr_row_major_Lower_template, "f32"
)
exo_ssyr_row_major_Lower_stride_any = rename(
    exo_ssyr_row_major_Lower_stride_any,
    exo_ssyr_row_major_Lower_stride_any.name() + "_stride_any",
)
f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_ssyr_row_major_Upper_stride_1 = schedule_interleave_syr_row_major_stride_1(
    syr_row_major_Upper_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_ssyr_row_major_Lower_stride_1 = schedule_interleave_syr_row_major_stride_1(
    syr_row_major_Lower_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dsyr_row_major_Upper_stride_any = specialize_syr(
    syr_row_major_Upper_template, "f64"
)
exo_dsyr_row_major_Upper_stride_any = rename(
    exo_dsyr_row_major_Upper_stride_any,
    exo_dsyr_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_dsyr_row_major_Lower_stride_any = specialize_syr(
    syr_row_major_Lower_template, "f64"
)
exo_dsyr_row_major_Lower_stride_any = rename(
    exo_dsyr_row_major_Lower_stride_any,
    exo_dsyr_row_major_Lower_stride_any.name() + "_stride_any",
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.fmadd_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dsyr_row_major_Upper_stride_1 = schedule_interleave_syr_row_major_stride_1(
    syr_row_major_Upper_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dsyr_row_major_Lower_stride_1 = schedule_interleave_syr_row_major_stride_1(
    syr_row_major_Lower_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)

entry_points = [
    exo_ssyr_row_major_Upper_stride_any,
    exo_ssyr_row_major_Upper_stride_1,
    exo_dsyr_row_major_Upper_stride_any,
    exo_dsyr_row_major_Upper_stride_1,
    exo_ssyr_row_major_Lower_stride_any,
    exo_ssyr_row_major_Lower_stride_1,
    exo_dsyr_row_major_Lower_stride_any,
    exo_dsyr_row_major_Lower_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
