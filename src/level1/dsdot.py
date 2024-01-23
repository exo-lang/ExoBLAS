from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *
from codegen_helpers import *
from parameters import Level_1_Params

### EXO_LOC ALGORITHM START ###
@proc
def dsdot_template(n: size, x: [f32][n], y: [f32][n], result: f64):
    d_dot: f64
    d_dot = 0.0
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_dot += d_x * d_y
    result = d_dot


@proc
def sdsdot_template(n: size, sb: f32, x: [f32][n], y: [f32][n], result: f32):
    d_result: f64
    d_dot: f64
    d_dot = 0.0
    for i in seq(0, n):
        d_x: f64
        d_x = x[i]
        d_y: f64
        d_y = y[i]
        d_dot += d_x * d_y
    d_result = d_dot
    d_result += sb
    result = d_result


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_dsdot_stride_1(proc, params, name):
    # TODO: This optimization strategy appears to be the wrong one from
    # benchmarking results. Need to investigate it why.

    proc = rename(proc, name)
    proc = proc.add_assertion("stride(x, 0) == 1")
    proc = proc.add_assertion("stride(y, 0) == 1")

    if params.mem_type is not AVX2:
        return proc

    main_loop = proc.find_loop("i")
    proc, cursors = auto_divide_loop(proc, main_loop, params.vec_width // 2, tail="cut")
    proc, cursors = auto_divide_loop(proc, main_loop, 2, tail="cut")
    proc = reorder_loops(proc, proc.find_loop("ioi"))
    proc = parallelize_reduction(proc, proc.find("d_dot += _"), params.mem_type, 3)
    proc = auto_stage_mem(proc, proc.find("x[_]"), "xReg", n_lifts=2)
    proc = auto_stage_mem(proc, proc.find("y[_]"), "yReg", n_lifts=2)
    proc = set_memory(proc, "xReg", params.mem_type)
    proc = set_memory(proc, "yReg", params.mem_type)
    proc = simplify(proc)
    proc = reorder_loops(proc, proc.find_loop("ii #1"))
    proc = scalar_to_simd(
        proc, proc.find_loop("ii #1"), params.vec_width // 2, params.mem_type, "f64"
    )
    proc = interleave_loop(proc, proc.find_loop("ioi"))
    proc, _ = auto_divide_loop(
        proc, proc.find_loop("ioo"), params.accumulators_count, tail="cut"
    )
    proc = parallelize_reduction(
        proc, proc.find("var0[_] += _"), params.mem_type, 3, True
    )
    instructions = [
        C.Machine.load_instr_f32,
        C.Machine.store_instr_f32,
        C.Machine.load_instr_f64,
        C.Machine.store_instr_f64,
        C.Machine.set_zero_instr_f64,
        C.Machine.fmadd_reduce_instr_f64,
        C.Machine.reduce_add_wide_instr_f64,
        C.Machine.assoc_reduce_add_instr_f64,
    ]

    proc = unfold_reduce(proc, proc.find("var1[_] += _ * _"))
    proc = unfold_reduce(proc, proc.find("var1[_] += _ * _"))
    proc = unfold_reduce(proc, proc.find("var0[_] += _ * _"))
    proc = unfold_reduce(proc, proc.find("var0[_] += _ * _"))
    proc = replace_all_stmts(proc, instructions)
    for i in range(0, 4):
        proc = replace(proc, proc.find_loop("ii"), C.Machine.convert_f32_lower_to_f64)
        proc = replace(proc, proc.find_loop("ii"), C.Machine.convert_f32_upper_to_f64)
    proc = interleave_loop(proc, proc.find_loop("iooi"))
    return proc


export_exo_proc(globals(), rename(dsdot_template, "exo_dsdot_stride_any"))
export_exo_proc(globals(), rename(sdsdot_template, "exo_sdsdot_stride_any"))
export_exo_proc(
    globals(),
    schedule_dsdot_stride_1(
        dsdot_template, Level_1_Params(precision="f32"), "exo_dsdot_stride_1"
    ),
)
export_exo_proc(
    globals(),
    schedule_dsdot_stride_1(
        sdsdot_template, Level_1_Params(precision="f32"), "exo_sdsdot_stride_1"
    ),
)

### EXO_LOC SCHEDULE END ###
