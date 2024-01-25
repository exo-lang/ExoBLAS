from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *
from blaslib import *
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
    proc = rename(proc, name)
    proc = proc.add_assertion("stride(x, 0) == 1")
    proc = proc.add_assertion("stride(y, 0) == 1")

    if params.mem_type is not AVX2:
        return proc
    instrs = params.instructions
    params.instructions = []
    proc = optimize_level_1(proc, proc.find_loop("i"), params)
    proc = replace_all_stmts(
        proc,
        [C.Machine.fused_load_cvt_f32_f64, C.Machine.prefix_fused_load_cvt_f32_f64],
    )
    proc = replace_all_stmts(proc, instrs)
    return proc


export_exo_proc(globals(), rename(dsdot_template, "exo_dsdot_stride_any"))
export_exo_proc(globals(), rename(sdsdot_template, "exo_sdsdot_stride_any"))
export_exo_proc(
    globals(),
    schedule_dsdot_stride_1(
        dsdot_template, Level_1_Params(precision="f64"), "exo_dsdot_stride_1"
    ),
)
export_exo_proc(
    globals(),
    schedule_dsdot_stride_1(
        sdsdot_template, Level_1_Params(precision="f64"), "exo_sdsdot_stride_1"
    ),
)

### EXO_LOC SCHEDULE END ###
