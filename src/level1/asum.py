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
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])


### EXO_LOC ALGORITHM END ###

### EXO_LOC SCHEDULE START ###
def schedule_asum_stride_1(asum, precision):
    asum = generate_stride_1_proc(asum, precision)
    if C.Machine.mem_type is not AVX2:
        return asum
    asum = optimize_level_1(asum, asum.find_loop("i"), precision, C.Machine, 7)
    return asum


template_sched_list = [
    (asum, schedule_asum_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        proc_stride_any = bind_builtins_args(
            proc_stride_any, proc_stride_any.body(), precision
        )
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(
            template,
            precision,
        )
        proc_stride_1 = bind_builtins_args(
            proc_stride_1, proc_stride_1.body(), precision
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
