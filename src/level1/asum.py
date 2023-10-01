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
    auto_divide_loop,
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
    asum = generate_stride_1_proc(asum, params.precision)

    VEC_W = params.vec_width
    INTERLEAVE_FACTOR = params.interleave_factor
    memory = params.mem_type
    instructions = params.instructions
    precision = params.precision

    if None in instructions:
        return asum

    asum = stage_mem(
        asum,
        asum.find_loop("i").body(),
        f"x[i]",
        "xReg",
    )

    asum, _ = parallelize_reduction(
        asum,
        asum.find_loop("i"),
        "result_",
        VEC_W,
        memory,
        precision,
    )

    asum = expand_dim(asum, "xReg", VEC_W, "ii")
    asum = lift_alloc(asum, "xReg")
    asum = fission(asum, asum.find("xReg[_] = _").after())

    asum = stage_expr(asum, asum.find("select(_)"), "selectReg")

    for buffer in ["xReg", "selectReg"]:
        asum = set_memory(asum, buffer, memory)
        asum = set_precision(asum, buffer, precision)

    asum, _ = parallelize_reduction(
        asum,
        asum.find_loop("io"),
        f"reg[0:{VEC_W}]",
        INTERLEAVE_FACTOR // 2,
        memory,
        precision,
    )

    asum = replace_all(asum, instructions)
    asum = unroll_loop(asum, asum.find_loop("ioi"))
    asum = interleave_execution(asum, asum.find_loop("ioi"), INTERLEAVE_FACTOR // 2)
    asum = interleave_execution(asum, asum.find_loop("ioo"), 2)
    asum = unroll_loop(asum, asum.find_loop("ioi"))

    asum = simplify(asum)

    return asum


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
