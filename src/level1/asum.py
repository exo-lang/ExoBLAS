from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    vectorize_to_loops,
    interleave_execution,
    parallelize_reduction,
    stage_expr,
)
from codegen_helpers import (
    specialize_precision,
    generate_stride_any_proc,
    export_exo_proc,
    generate_stride_1_proc,
    bind_builtins_args,
)
from parameters import Level_1_Params

### EXO_LOC ALGORITHM START ###
@proc
def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
    result = 0.0
    for i in seq(0, n):
        result += select(0.0, x[i], x[i], -x[i])


### EXO_LOC ALGORITHM END ###

### EXO_LOC SCHEDULE START ###
def schedule_asum_stride_1(asum, params):
    simple_stride_1 = generate_stride_1_proc(asum, params.precision)

    VEC_W = params.vec_width
    INTERLEAVE_FACTOR = params.interleave_factor
    memory = params.mem_type
    instructions = params.instructions
    precision = params.precision

    if None in instructions:
        return simple_stride_1

    simple_stride_1 = stage_mem(
        simple_stride_1,
        simple_stride_1.find_loop("i").body(),
        f"x[i]",
        "xReg",
    )

    simple_stride_1, _ = parallelize_reduction(
        simple_stride_1,
        simple_stride_1.find_loop("i"),
        "result",
        VEC_W,
        memory,
        precision,
    )

    simple_stride_1 = expand_dim(simple_stride_1, "xReg", VEC_W, "ii")
    simple_stride_1 = lift_alloc(simple_stride_1, "xReg")
    simple_stride_1 = fission(
        simple_stride_1, simple_stride_1.find("xReg[_] = _").after()
    )

    simple_stride_1 = stage_expr(
        simple_stride_1, simple_stride_1.find("select(_)"), "selectReg"
    )

    for buffer in ["xReg", "selectReg"]:
        simple_stride_1 = set_memory(simple_stride_1, buffer, memory)
        simple_stride_1 = set_precision(simple_stride_1, buffer, precision)

    simple_stride_1, _ = parallelize_reduction(
        simple_stride_1,
        simple_stride_1.find_loop("io"),
        f"reg[0:{VEC_W}]",
        INTERLEAVE_FACTOR // 2,
        memory,
        precision,
    )

    simple_stride_1 = replace_all(simple_stride_1, instructions)
    simple_stride_1 = unroll_loop(simple_stride_1, simple_stride_1.find_loop("ioi"))
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("ioi"), INTERLEAVE_FACTOR // 2
    )
    simple_stride_1 = interleave_execution(
        simple_stride_1, simple_stride_1.find_loop("ioo"), 2
    )
    simple_stride_1 = unroll_loop(simple_stride_1, simple_stride_1.find_loop("ioi"))

    simple_stride_1 = simplify(simple_stride_1)

    return simple_stride_1


INTERLEAVE_FACTOR = 8

#################################################
# Generate specialized kernels for f32 precision
#################################################

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
        proc_stride_1 = sched(template, Level_1_Params(precision=precision))
        proc_stride_1 = bind_builtins_args(
            proc_stride_1, proc_stride_1.body(), precision
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
