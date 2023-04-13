from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    apply_to_block,
    hoist_stmt,
)


### EXO_LOC ALGORITHM START ###
@proc
def rot_template(n: size, x: [R][n], y: [R][n], c: R, s: R):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = c * xReg + s * y[i]
        y[i] = -s * xReg + c * y[i]
### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(rot_template, "exo_" + prefix + "rot")
    for arg in ["x", "y", "c", "s", "xReg"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy


def schedule_rot_stride_1(VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision):
    simple_stride_1 = specialize_precision(precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    loop_cursor = simple_stride_1.find_loop("i")
    simple_stride_1 = bind_expr(
        simple_stride_1, simple_stride_1.find("y[_]", many=True), "yReg", cse=True
    )
    simple_stride_1 = bind_expr(
        simple_stride_1, simple_stride_1.find("s", many=True), "sReg", cse=True
    )
    simple_stride_1 = bind_expr(
        simple_stride_1, simple_stride_1.find("c", many=True), "cReg", cse=True
    )
    simple_stride_1 = vectorize(simple_stride_1, loop_cursor, VEC_W, memory, precision)
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("io"), INTERLEAVE_FACTOR
    )

    regs_to_hoist = ["reg3", "reg2", "sReg", "reg11", "cReg", "reg4"]
    stmts_to_hoist = ["c", "s", "reg2[_]", "-sReg[_]", "reg3[_]", "cReg[_]"]
    for reg in regs_to_hoist:
        simple_stride_1 = lift_alloc(simple_stride_1, reg)
    for stmt in stmts_to_hoist:
        loop = simple_stride_1.find(stmt).parent().parent()
        simple_stride_1 = hoist_stmt(simple_stride_1, loop)

    simple_stride_1 = replace_all(simple_stride_1, instructions)

    # Set temporary variables in tail loop
    for stmt in simple_stride_1.find_loop("ii").body():
        if isinstance(stmt, pc.AllocCursor):
            simple_stride_1 = set_precision(simple_stride_1, stmt, precision)

    return simple_stride_1


INTERLEAVE_FACTOR = C.Machine.vec_units

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_srot_stride_any = specialize_precision("f32")
exo_srot_stride_any = rename(
    exo_srot_stride_any, exo_srot_stride_any.name() + "_stride_any"
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.add_instr_f32,
    C.Machine.reg_copy_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
    C.Machine.sign_instr_f32,
]

if None not in f32_instructions:
    exo_srot_stride_1 = schedule_rot_stride_1(
        C.Machine.vec_width,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
else:
    exo_srot_stride_1 = specialize_precision("f32")
    exo_srot_stride_1 = rename(
        exo_srot_stride_1, exo_srot_stride_1.name() + "_stride_1"
    )

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_drot_stride_any = specialize_precision("f64")
exo_drot_stride_any = rename(
    exo_drot_stride_any, exo_drot_stride_any.name() + "_stride_any"
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.mul_instr_f64,
    C.Machine.add_instr_f64,
    C.Machine.reg_copy_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
    C.Machine.sign_instr_f64,
]

if None not in f64_instructions:
    exo_drot_stride_1 = schedule_rot_stride_1(
        C.Machine.vec_width // 2,
        INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
else:
    exo_drot_stride_1 = specialize_precision("f64")
    exo_drot_stride_1 = rename(
        exo_drot_stride_1, exo_drot_stride_1.name() + "_stride_1"
    )
### EXO_LOC SCHEDULE END ###

entry_points = [
    exo_srot_stride_any,
    exo_srot_stride_1,
    exo_drot_stride_any,
    exo_drot_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
