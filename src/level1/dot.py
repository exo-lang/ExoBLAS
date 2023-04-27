from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import (
    apply_to_block,
    hoist_stmt,
)
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params


### EXO_LOC ALGORITHM START ###
@proc
def dot_template(n: size, x: [R][n], y: [R][n], result: R):
    result = 0.0
    for i in seq(0, n):
        result += x[i] * y[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_dot_stride_1(dot, params):
    dot = generate_stride_1_proc(dot, params.precision)
    main_loop = dot.find_loop("i")
    dot = blas_vectorize(dot, main_loop, params)
    return simplify(dot)


template_sched_list = [
    (dot_template, schedule_dot_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, Level_1_Params(precision=precision))
        export_exo_proc(globals(), proc_stride_1)
### EXO_LOC SCHEDULE END ###
