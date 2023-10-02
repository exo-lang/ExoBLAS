from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    interleave_execution,
    parallelize_reduction,
    stage_expr,
    auto_stage_mem,
)
from codegen_helpers import (
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

    if None in params.instructions:
        return asum

    loop = asum.find_loop("i")

    asum, _ = parallelize_reduction(
        asum,
        loop,
        "result_",
        params.vec_width,
        params.mem_type,
        params.precision,
        tail="guard",
    )

    loop = asum.forward(loop)
    asum = cut_loop(asum, loop, FormattedExprStr("_ - 1", loop.hi()))
    asum = remove_if(asum, loop.body()[0].body()[0])
    asum = auto_stage_mem(asum, asum.find("x[_]"), "xReg")
    asum = stage_expr(asum, asum.find("select(_)"), "selectReg")
    asum = simplify(asum)

    tail_select = asum.find("select(_) #1")
    args = [tail_select.args()[1], tail_select.args()[2], tail_select.args()[3].arg()]
    asum = stage_expr(asum, args, "xRegTail", n_lifts=2)
    asum = stage_expr(asum, tail_select, "selectRegTail", n_lifts=2)

    for buffer in ["xReg", "selectReg", "xRegTail", "selectRegTail"]:
        asum = set_memory(asum, buffer, params.mem_type)
        asum = set_precision(asum, buffer, params.precision)

    asum, _ = parallelize_reduction(
        asum,
        asum.find_loop("io"),
        f"reg[0:{params.vec_width}]",
        params.accumulators_count,
        params.mem_type,
        params.precision,
    )

    asum = replace_all(asum, params.instructions)

    asum = unroll_loop(asum, asum.find_loop("ioi"))
    asum = interleave_execution(asum, asum.find_loop("ioi"), params.accumulators_count)
    asum = interleave_execution(
        asum,
        asum.find_loop("ioo"),
        params.interleave_factor // params.accumulators_count,
    )
    asum = unroll_loop(asum, asum.find_loop("ioi"))
    asum = simplify(asum)

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
            Level_1_Params(
                precision=precision, interleave_factor=7, accumulators_count=7
            ),
        )
        proc_stride_1 = bind_builtins_args(
            proc_stride_1, proc_stride_1.body(), precision
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###
