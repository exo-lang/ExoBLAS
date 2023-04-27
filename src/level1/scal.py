from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import (
    vectorize_to_loops,
    interleave_execution,
    hoist_stmt,
    apply_to_block,
)


### EXO_LOC ALGORITHM START ###
@proc
def scal_template(n: size, alpha: R, x: [R][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]


@proc
def scal_template_alpha_0(n: size, x: [R][n]):
    for i in seq(0, n):
        x[i] = 0.0


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_scal(precision, alpha):
    prefix = "s" if precision == "f32" else "d"
    specialized_scal = scal_template if alpha != 0 else scal_template_alpha_0
    specialized_scal_name = specialized_scal.name()
    specialized_scal_name = specialized_scal_name.replace("_template", "")
    specialized_scal = rename(specialized_scal, "exo_" + prefix + specialized_scal_name)

    args = ["x"]
    if alpha != 0:
        args.append("alpha")

    for arg in args:
        specialized_scal = set_precision(specialized_scal, arg, precision)
    return specialized_scal


def schedule_scal_stride_1(
    VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision, alpha
):
    simple_stride_1 = specialize_scal(precision, alpha)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")

    main_loop = simple_stride_1.find_loop("i")
    simple_stride_1 = vectorize_to_loops(
        simple_stride_1, main_loop, VEC_W, memory, precision
    )
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("io"), INTERLEAVE_FACTOR
    )
    simple_stride_1 = apply_to_block(
        simple_stride_1, simple_stride_1.find_loop("ioo").body(), hoist_stmt
    )
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    return simple_stride_1


#################################################
# Generate specialized kernels for f32 precision
#################################################

INTERLEAVE_FACTOR = C.Machine.vec_units * 2

exo_sscal_stride_any = specialize_scal("f32", None)
exo_sscal_stride_any = rename(
    exo_sscal_stride_any, exo_sscal_stride_any.name() + "_stride_any"
)
exo_sscal_alpha_0_stride_any = specialize_scal("f32", 0)
exo_sscal_alpha_0_stride_any = rename(
    exo_sscal_alpha_0_stride_any, exo_sscal_alpha_0_stride_any.name() + "_stride_any"
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
    C.Machine.set_zero_instr_f32,
]

if None not in f32_instructions:
    exo_sscal_stride_1 = schedule_scal_stride_1(
        C.Machine.vec_width,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
        None,
    )
    exo_sscal_alpha_0_stride_1 = schedule_scal_stride_1(
        C.Machine.vec_width,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
        0,
    )
else:
    exo_sscal_stride_1 = specialize_scal("f32", None)
    exo_sscal_stride_1 = rename(
        exo_sscal_stride_1, exo_sscal_stride_1.name() + "_stride_1"
    )
    exo_sscal_alpha_0_stride_1 = specialize_scal("f32", 0)
    exo_sscal_alpha_0_stride_1 = rename(
        exo_sscal_alpha_0_stride_1, exo_sscal_alpha_0_stride_1.name() + "_stride_1"
    )

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dscal_stride_any = specialize_scal("f64", None)
exo_dscal_stride_any = rename(
    exo_dscal_stride_any, exo_dscal_stride_any.name() + "_stride_any"
)
exo_dscal_alpha_0_stride_any = specialize_scal("f64", 0)
exo_dscal_alpha_0_stride_any = rename(
    exo_dscal_alpha_0_stride_any, exo_dscal_alpha_0_stride_any.name() + "_stride_any"
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
    C.Machine.set_zero_instr_f64,
]

if None not in f64_instructions:
    exo_dscal_stride_1 = schedule_scal_stride_1(
        C.Machine.vec_width // 2,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
        None,
    )
    exo_dscal_alpha_0_stride_1 = schedule_scal_stride_1(
        C.Machine.vec_width // 2,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
        0,
    )
else:
    exo_dscal_stride_1 = specialize_scal("f64", None)
    exo_dscal_stride_1 = rename(
        exo_dscal_stride_1, exo_dscal_stride_1.name() + "_stride_1"
    )
    exo_dscal_alpha_0_stride_1 = specialize_scal("f64", 0)
    exo_dscal_alpha_0_stride_1 = rename(
        exo_dscal_alpha_0_stride_1, exo_dscal_alpha_0_stride_1.name() + "_stride_1"
    )
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_sscal_stride_any,
    exo_sscal_stride_1,
    exo_sscal_alpha_0_stride_1,
    exo_sscal_alpha_0_stride_any,
    exo_dscal_stride_any,
    exo_dscal_stride_1,
    exo_dscal_alpha_0_stride_1,
    exo_dscal_alpha_0_stride_any,
]


if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
