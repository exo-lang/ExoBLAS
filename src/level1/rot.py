from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import *
from blaslib import *
from codegen_helpers import (
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params

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
def schedule_rot_stride_1(rot, params):
    rot = generate_stride_1_proc(rot, params.precision)
    loop_cursor = rot.find_loop("i")
    rot = bind_expr(rot, rot.find("y[_]", many=True), "yReg")
    rot = set_precision(rot, "yReg", params.precision)
    rot = bind_expr(rot, rot.find("s_", many=True), "sReg")
    rot = set_precision(rot, "sReg", params.precision)
    rot = bind_expr(rot, rot.find("c_", many=True), "cReg")
    rot = set_precision(rot, "cReg", params.precision)
    rot = optimize_level_1(rot, loop_cursor, params)
    return rot


template_sched_list = [
    (rot_template, schedule_rot_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, Level_1_Params(precision=precision))
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
