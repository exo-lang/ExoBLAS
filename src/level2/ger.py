from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


### EXO_LOC ALGORITHM START ###
@proc
def ger_row_major_template(
    m: size, n: size, alpha: R, x: [R][m], y: [R][n], A: [R][m, n]
):
    assert stride(A, 1) == 1

    for i in seq(0, m):
        for j in seq(0, n):
            A[i, j] += alpha * x[i] * y[j]


@proc
def ger_row_major_template_alpha_1(
    m: size, n: size, alpha: R, x: [R][m], y: [R][n], A: [R][m, n]
):
    assert stride(A, 1) == 1

    for i in seq(0, m):
        for j in seq(0, n):
            A[i, j] += x[i] * y[j]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_ger(precision, alpha):
    prefix = "s" if precision == "f32" else "d"
    specialized = (
        ger_row_major_template if alpha != 1 else ger_row_major_template_alpha_1
    )
    name = specialized.name().replace("_template", "")
    specialized = rename(specialized, "exo_" + prefix + name)

    args = ["alpha", "x", "y", "A"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return simplify(specialized)


def schedule_ger_row_major_stride_1(
    VEC_W, ROW_INTERLEAVE_FACTOR, memory, instructions, precision, alpha
):
    stride_1 = specialize_ger(precision, alpha)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    stride_1 = stage_mem(stride_1, stride_1.body(), "alpha", "alpha_")

    return simplify(stride_1)


#################################################
# Kernel Parameters
#################################################

ROW_INTERLEAVE_FACTOR = C.Machine.vec_units

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sger_row_major_stride_any = specialize_ger("f32", None)
exo_sger_row_major_stride_any = rename(
    exo_sger_row_major_stride_any, exo_sger_row_major_stride_any.name() + "_stride_any"
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_reduce_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_sger_row_major_stride_1 = schedule_ger_row_major_stride_1(
    C.Machine.f32_vec_width,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
    None,
)
exo_sger_row_major_alpha_1_stride_1 = schedule_ger_row_major_stride_1(
    C.Machine.f32_vec_width,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
    1,
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dger_row_major_stride_any = specialize_ger("f64", None)
exo_dger_row_major_stride_any = rename(
    exo_dger_row_major_stride_any, exo_dger_row_major_stride_any.name() + "_stride_any"
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.fmadd_reduce_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dger_row_major_stride_1 = schedule_ger_row_major_stride_1(
    C.Machine.f32_vec_width // 2,
    ROW_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
    None,
)
### EXO_LOC SCHEDULE END ###


entry_points = [
    exo_sger_row_major_stride_any,
    exo_sger_row_major_stride_1,
    exo_dger_row_major_stride_any,
    exo_dger_row_major_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
