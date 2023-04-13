from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import vectorize, interleave_execution


@proc
def copy_template(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] = x[i]


def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(copy_template, "exo_" + prefix + "copy")
    for arg in ["x", "y"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy


def schedule_scopy_stride_1(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    simple_stride_1 = vectorize(
        simple_stride_1, simple_stride_1.find_loop("i"), VEC_W, memory, precision
    )
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("io"), INTERLEAVE_FACTOR
    )
    simple_stride_1 = replace_all(simple_stride_1, instructions)

    return simplify(simple_stride_1)


INTERLEAVE_FACTOR = C.Machine.vec_units * 2

f32_instructions = [C.Machine.load_instr_f32, C.Machine.store_instr_f32]

exo_scopy_stride_1 = schedule_scopy_stride_1(
    C.Machine.vec_width, INTERLEAVE_FACTOR, C.Machine.mem_type, f32_instructions, "f32"
)
exo_scopy_stride_any = specialize_precision("f32")
exo_scopy_stride_any = rename(
    exo_scopy_stride_any, exo_scopy_stride_any.name() + "_stride_any"
)

f64_instructions = [C.Machine.load_instr_f64, C.Machine.store_instr_f64]

if None not in f64_instructions:
    exo_dcopy_stride_1 = schedule_scopy_stride_1(
        C.Machine.vec_width // 2,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
else:
    exo_dcopy_stride_1 = specialize_precision("f64")
    exo_dcopy_stride_1 = rename(
        exo_dcopy_stride_1, exo_dcopy_stride_1.name() + "_stride_1"
    )

exo_dcopy_stride_any = specialize_precision("f64")
exo_dcopy_stride_any = rename(
    exo_dcopy_stride_any, exo_dcopy_stride_any.name() + "_stride_any"
)

entry_points = [
    exo_scopy_stride_1,
    exo_dcopy_stride_1,
    exo_scopy_stride_any,
    exo_dcopy_stride_any,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
