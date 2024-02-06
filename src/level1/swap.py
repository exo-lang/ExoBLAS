from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from blaslib import *
from codegen_helpers import *

### EXO_LOC ALGORITHM START ###
@proc
def swap_template(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        tmp: R
        tmp = x[i]
        x[i] = y[i]
        y[i] = tmp


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_swap(swap, precision):
    swap = generate_stride_1_proc(swap, precision)
    main_loop = swap.find_loop("i")
    swap = optimize_level_1(swap, main_loop, precision, C.Machine, 4)
    return simplify(swap)


template_sched_list = [
    (swap_template, schedule_swap),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, precision)
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
