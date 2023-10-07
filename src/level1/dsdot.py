from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from blas_composed_schedules import blas_vectorize
from composed_schedules import (
    parallelize_reduction,
    auto_divide_loop,
    auto_stage_mem,
    vectorize_to_loops,
    interleave_execution,
)
from codegen_helpers import (
    export_exo_proc,
)
from parameters import Level_1_Params

### EXO_LOC ALGORITHM START ###
@proc
def dsdot_template(n: size, x: [f32][n], y: [f32][n], result: f64):
    d_result: f64
    d_result = 0.0
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_result += d_x * d_y
    result = d_result


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_dsdot_stride_1(params):
    dsdot = rename(dsdot_template, "exo_dsdot_stride_1")
    dsdot = dsdot.add_assertion("stride(x, 0) == 1")
    dsdot = dsdot.add_assertion("stride(y, 0) == 1")

    if params.mem_type is not AVX2:
        return dsdot

    main_loop = dsdot.find_loop("i")
    dsdot, cursors = auto_divide_loop(
        dsdot, main_loop, params.vec_width // 2, tail="cut"
    )
    dsdot, cursors = auto_divide_loop(dsdot, main_loop, 2, tail="cut")
    dsdot = reorder_loops(dsdot, dsdot.find_loop("ioi"))
    dsdot, _ = parallelize_reduction(
        dsdot, main_loop, "d_result", params.vec_width // 2, params.mem_type, "f64"
    )
    dsdot = auto_stage_mem(dsdot, dsdot.find("x[_]"), "xReg", n_lifts=2)
    dsdot = auto_stage_mem(dsdot, dsdot.find("y[_]"), "yReg", n_lifts=2)
    dsdot = set_memory(dsdot, "xReg", params.mem_type)
    dsdot = set_memory(dsdot, "yReg", params.mem_type)
    dsdot = simplify(dsdot)
    dsdot = reorder_loops(dsdot, dsdot.find_loop("ii #1"))
    dsdot = vectorize_to_loops(
        dsdot, dsdot.find_loop("ii #1"), params.vec_width // 2, params.mem_type, "f64"
    )
    dsdot = interleave_execution(dsdot, dsdot.find_loop("ioi"), 2)
    dsdot, _ = parallelize_reduction(
        dsdot,
        dsdot.find_loop("ioo"),
        f"reg[0:{params.vec_width // 2}]",
        params.accumulators_count,
        params.mem_type,
        "f64",
        tail="cut",
    )
    instructions = [
        C.Machine.load_instr_f32,
        C.Machine.store_instr_f32,
        C.Machine.load_instr_f64,
        C.Machine.store_instr_f64,
        C.Machine.set_zero_instr_f64,
        C.Machine.fmadd_instr_f64,
        C.Machine.reduce_add_wide_instr_f64,
        C.Machine.assoc_reduce_add_instr_f64,
    ]

    dsdot = replace_all(dsdot, instructions)
    for i in range(0, 4):
        dsdot = replace(
            dsdot, dsdot.find_loop("ii"), C.Machine.convert_f32_lower_to_f64
        )
        dsdot = replace(
            dsdot, dsdot.find_loop("ii"), C.Machine.convert_f32_upper_to_f64
        )
    dsdot = unroll_loop(dsdot, "iooi")
    dsdot = interleave_execution(
        dsdot, dsdot.find_loop("iooi"), params.accumulators_count
    )
    dsdot = unroll_loop(dsdot, "iooi")
    return dsdot


export_exo_proc(globals(), rename(dsdot_template, "exo_dsdot_stride_any"))
export_exo_proc(globals(), schedule_dsdot_stride_1(Level_1_Params(precision="f32")))

### EXO_LOC SCHEDULE END ###
