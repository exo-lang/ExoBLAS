from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

import exo_blas_config as C
from composed_schedules import *
from blaslib import *
from codegen_helpers import *
from parameters import *

### EXO_LOC ALGORITHM START ###
@proc
def axpy_template(n: size, alpha: R, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] += alpha * x[i]


@proc
def axpy_template_alpha_1(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] += x[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_axpy_stride_1(axpy, params):
    axpy = generate_stride_1_proc(axpy, params.precision)
    main_loop = axpy.find_loop("i")
    axpy = optimize_level_1(axpy, main_loop, params)
    return simplify(axpy)


template_sched_list = [
    (axpy_template, schedule_axpy_stride_1),
    (axpy_template_alpha_1, schedule_axpy_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, Level_1_Params(precision=precision))
        export_exo_proc(globals(), proc_stride_1)
### EXO_LOC SCHEDULE END ###
