from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import vectorize


### EXO_LOC ALGORITHM START ###
@proc
def dot_template(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(dot_template, "exo_" + prefix + "dot")
    for arg in ["x", "y", "result"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy


def schedule_dot_stride_1_interleaved(
    VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    loop_cursor = simple_stride_1.find_loop("i")
    simple_stride_1 = vectorize(
        simple_stride_1,
        loop_cursor,
        VEC_W,
        INTERLEAVE_FACTOR,
        INTERLEAVE_FACTOR,
        memory,
        precision,
        instructions,
    )
    return simplify(simple_stride_1)


INTERLEAVE_FACTOR = C.Machine.vec_units * 2

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sdot_stride_any = specialize_precision("f32")
exo_sdot_stride_any = rename(
    exo_sdot_stride_any, exo_sdot_stride_any.name() + "_stride_any"
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.assoc_reduce_add_instr_f32,
    C.Machine.set_zero_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.reg_copy_instr_f32,
]

if None not in f32_instructions:
    exo_sdot_stride_1 = schedule_dot_stride_1_interleaved(
        C.Machine.vec_width,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
else:
    exo_sdot_stride_1 = specialize_precision("f32")
    exo_sdot_stride_1 = rename(
        exo_sdot_stride_1, exo_sdot_stride_1.name() + "_stride_1"
    )

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_ddot_stride_any = specialize_precision("f64")
exo_ddot_stride_any = rename(
    exo_ddot_stride_any, exo_ddot_stride_any.name() + "_stride_any"
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.assoc_reduce_add_instr_f64,
    C.Machine.set_zero_instr_f64,
    C.Machine.fmadd_instr_f64,
    C.Machine.reg_copy_instr_f64,
]

if None not in f64_instructions:
    exo_ddot_stride_1 = schedule_dot_stride_1_interleaved(
        C.Machine.vec_width // 2,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
else:
    exo_ddot_stride_1 = specialize_precision("f64")
    exo_ddot_stride_1 = rename(
        exo_ddot_stride_1, exo_ddot_stride_1.name() + "_stride_1"
    )
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_sdot_stride_any,
    exo_sdot_stride_1,
    exo_ddot_stride_any,
    exo_ddot_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
