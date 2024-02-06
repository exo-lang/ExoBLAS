from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

from blaslib import *
from codegen_helpers import *
import exo_blas_config as C


### EXO_LOC ALGORITHM START ###
@proc
def copy_template(n: size, x: [R][n], y: [R][n]):
    for i in seq(0, n):
        y[i] = x[i]


### EXO_LOC ALGORITHM END ###

### EXO_LOC SCHEDULE START ###
def schedule_copy(exo_copy, precision):
    exo_copy = generate_stride_1_proc(exo_copy, precision)
    main_loop = exo_copy.find_loop("i")
    exo_copy = optimize_level_1(exo_copy, main_loop, precision, C.Machine, 4)
    return simplify(exo_copy)


template_sched_list = [
    (copy_template, schedule_copy),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(template, precision)
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
