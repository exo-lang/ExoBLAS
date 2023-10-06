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
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
)
from parameters import Level_1_Params

### EXO_LOC ALGORITHM START ###
@proc
def scal_template(n: size, alpha: R, x: [R][n]):
    for i in seq(0, n):
        x[i] = alpha * x[i]


@proc
def scal_template_alpha_0(n: size, x: [R][n]):
    for i in seq(0, n):
        x[i] = 0.0


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def schedule_scal_stride_1(scal, params):
    scal = generate_stride_1_proc(scal, params.precision)
    main_loop = scal.find_loop("i")
    scal = blas_vectorize(scal, main_loop, params)
    print(scal)
    main_loop = scal.find_loop("ioo")
    scal = add_unsafe_guard(
        scal,
        main_loop.as_block(),
        FormattedExprStr("_ < _", main_loop.lo(), main_loop.hi()),
    )
    main_loop = scal.find_loop("ioo")
    scal = apply_to_block(scal, main_loop.body(), hoist_stmt)
    middle_loop = scal.find_loop("ioi")
    scal = add_unsafe_guard(
        scal,
        middle_loop.as_block(),
        FormattedExprStr("_ < _", middle_loop.lo(), middle_loop.hi()),
    )
    middle_loop = scal.find_loop("ioi")
    scal = apply_to_block(scal, middle_loop.body(), hoist_stmt)
    return simplify(scal)


template_sched_list = [
    (scal_template, schedule_scal_stride_1),
    (scal_template_alpha_0, schedule_scal_stride_1),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(
            template,
            Level_1_Params(
                precision=precision, accumulators_count=1, interleave_factor=4
            ),
        )
        export_exo_proc(globals(), proc_stride_1)
### EXO_LOC SCHEDULE END ###
