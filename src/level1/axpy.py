from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
import exo.API_cursors as pc

import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    apply_to_block,
    hoist_stmt,
)
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
)

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
def schedule_axpy_stride_1(
    axpy, VEC_W, INTERLEAVE_FACTOR, memory, instructions, precision
):
    simple_stride_1 = specialize_precision(axpy, precision)
    simple_stride_1 = rename(simple_stride_1, simple_stride_1.name() + "_stride_1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(x, 0) == 1")
    simple_stride_1 = simple_stride_1.add_assertion("stride(y, 0) == 1")

    main_loop = simple_stride_1.find_loop("i")
    simple_stride_1 = vectorize(simple_stride_1, main_loop, VEC_W, memory, precision)
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("io"), INTERLEAVE_FACTOR
    )
    simple_stride_1 = apply_to_block(
        simple_stride_1, simple_stride_1.find_loop("ioo").body(), hoist_stmt
    )
    simple_stride_1 = replace_all(simple_stride_1, instructions)
    return simplify(simple_stride_1)


#################################################
# Generate Entry Points
#################################################

template_sched_list = [
    (axpy_template, schedule_axpy_stride_1),
    (axpy_template_alpha_1, schedule_axpy_stride_1),
]

VECTORIZATION_INTERLEAVE_FACTOR = C.Machine.vec_units * 2

for vec_width, precision in (
    (C.Machine.vec_width, "f32"),
    (C.Machine.vec_width // 2, "f64"),
):
    instructions = [
        C.Machine[f"load_instr_{precision}"],
        C.Machine[f"store_instr_{precision}"],
        C.Machine[f"fmadd_instr_{precision}"],
        C.Machine[f"reg_copy_instr_{precision}"],
        C.Machine[f"set_zero_instr_{precision}"],
        C.Machine[f"reduce_add_wide_instr_{precision}"],
        C.Machine[f"broadcast_scalar_instr_{precision}"],
        C.Machine[f"assoc_reduce_add_instr_{precision}"],
    ]

    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(
            template,
            vec_width,
            VECTORIZATION_INTERLEAVE_FACTOR,
            C.Machine.mem_type,
            instructions,
            precision,
        )
        export_exo_proc(globals(), proc_stride_1)
### EXO_LOC SCHEDULE END ###
