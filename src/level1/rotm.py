from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import *
from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def rotm_template_flag_neg_one(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = H[0, 0] * xReg + H[0, 1] * y[i]
        y[i] = H[1, 0] * xReg + H[1, 1] * y[i]


@proc
def rotm_template_flag_zero(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = xReg + H[0, 1] * y[i]
        y[i] = H[1, 0] * xReg + y[i]


@proc
def rotm_template_flag_one(n: size, x: [R][n], y: [R][n], H: R[2, 2]):
    for i in seq(0, n):
        xReg: R
        xReg = x[i]
        x[i] = H[0, 0] * xReg + y[i]
        y[i] = -xReg + H[1, 1] * y[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###


def schedule_rotm_stride_1(rotm, precision):
    rotm = generate_stride_1_proc(rotm, precision)
    loop_cursor = rotm.find_loop("i")
    rotm = optimize_level_1(rotm, loop_cursor, precision, C.Machine, 4)
    return rotm


template_sched_list = [
    (rotm_template_flag_neg_one, schedule_rotm_stride_1),
    (rotm_template_flag_zero, schedule_rotm_stride_1),
    (rotm_template_flag_one, schedule_rotm_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, precision)
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
