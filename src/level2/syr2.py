from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


### EXO_LOC ALGORITHM START ###
@proc
def syr2_row_major_Upper_template(
    n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]
):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, n - i):
            A[i, i + j] += alpha * x[i] * y[i + j] + alpha * y[i] * x[i + j]


@proc
def syr2_row_major_Lower_template(
    n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]
):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, i + 1):
            A[i, j] += alpha * x[i] * y[j] + alpha * y[i] * x[j]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_syr2(syr2, precision):
    prefix = "s" if precision == "f32" else "d"
    name = syr2.name()
    name = name.replace("_template", "")
    specialized = rename(syr2, "exo_" + prefix + name)

    args = ["alpha", "x", "y", "A"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return simplify(specialized)


def schedule_interleave_syr2_row_major_stride_1(
    syr2, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    stride_1 = specialize_syr2(syr2, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    return simplify(stride_1)


#################################################
# Kernel Parameters
#################################################

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_ssyr2_row_major_Upper_stride_any = specialize_syr2(
    syr2_row_major_Upper_template, "f32"
)
exo_ssyr2_row_major_Upper_stride_any = rename(
    exo_ssyr2_row_major_Upper_stride_any,
    exo_ssyr2_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_ssyr2_row_major_Lower_stride_any = specialize_syr2(
    syr2_row_major_Lower_template, "f32"
)
exo_ssyr2_row_major_Lower_stride_any = rename(
    exo_ssyr2_row_major_Lower_stride_any,
    exo_ssyr2_row_major_Lower_stride_any.name() + "_stride_any",
)
f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_reduce_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_ssyr2_row_major_Upper_stride_1 = schedule_interleave_syr2_row_major_stride_1(
    syr2_row_major_Upper_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_ssyr2_row_major_Lower_stride_1 = schedule_interleave_syr2_row_major_stride_1(
    syr2_row_major_Lower_template,
    C.Machine.vec_width,
    1,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dsyr2_row_major_Upper_stride_any = specialize_syr2(
    syr2_row_major_Upper_template, "f64"
)
exo_dsyr2_row_major_Upper_stride_any = rename(
    exo_dsyr2_row_major_Upper_stride_any,
    exo_dsyr2_row_major_Upper_stride_any.name() + "_stride_any",
)
exo_dsyr2_row_major_Lower_stride_any = specialize_syr2(
    syr2_row_major_Lower_template, "f64"
)
exo_dsyr2_row_major_Lower_stride_any = rename(
    exo_dsyr2_row_major_Lower_stride_any,
    exo_dsyr2_row_major_Lower_stride_any.name() + "_stride_any",
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.fmadd_reduce_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dsyr2_row_major_Upper_stride_1 = schedule_interleave_syr2_row_major_stride_1(
    syr2_row_major_Upper_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dsyr2_row_major_Lower_stride_1 = schedule_interleave_syr2_row_major_stride_1(
    syr2_row_major_Lower_template,
    C.Machine.vec_width // 2,
    1,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_ssyr2_row_major_Upper_stride_any,
    exo_ssyr2_row_major_Upper_stride_1,
    exo_dsyr2_row_major_Upper_stride_any,
    exo_dsyr2_row_major_Upper_stride_1,
    exo_ssyr2_row_major_Lower_stride_any,
    exo_ssyr2_row_major_Lower_stride_1,
    exo_dsyr2_row_major_Lower_stride_any,
    exo_dsyr2_row_major_Lower_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
