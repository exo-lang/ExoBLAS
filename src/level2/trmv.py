from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
)
from codegen_helpers import (
    generate_stride_any_proc,
    export_exo_proc,
)


### EXO_LOC ALGORITHM START ###
@proc
def trmv_row_major_Upper_NonTrans_Unit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
        xCopy[n - i - 1] = dot

    for l in seq(0, n):
        x[l] += xCopy[l]


@proc
def trmv_row_major_Upper_NonTrans_NonUnit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, n - i):
            dot += A[i, i + j] * x[i + j]
        x[i] = dot


@proc
def trmv_row_major_Lower_NonTrans_Unit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[i, j] * x[j]
        xCopy[i] = dot

    for l in seq(0, n):
        x[l] += xCopy[l]


@proc
def trmv_row_major_Lower_NonTrans_NonUnit_template(n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, n - i):
            dot += A[n - i - 1, j] * x[j]
        x[n - i - 1] = dot


@proc
def trmv_row_major_Upper_Trans_Unit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]

    for i in seq(0, n):
        x[i] += xCopy[i]


@proc
def trmv_row_major_Upper_Trans_NonUnit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        xCopy[i] += A[i, i] * x[i]
        for j in seq(0, n - i - 1):
            xCopy[i + j + 1] += A[i, i + j + 1] * x[i]

    for i in seq(0, n):
        x[i] = xCopy[i]


@proc
def trmv_row_major_Lower_Trans_Unit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]

    for i in seq(0, n):
        x[i] += xCopy[i]


@proc
def trmv_row_major_Lower_Trans_NonUnit_template(
    n: size, x: [R][n], A: [R][n, n], Diag: size
):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for i in seq(0, n):
        xCopy[i] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]
        xCopy[i] += A[i, i] * x[i]

    for i in seq(0, n):
        x[i] = xCopy[i]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_trmv(trmv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = trmv.name()
    name = name.replace("_template", "")
    specialized = rename(trmv, "exo_" + prefix + name)

    args = ["x", "A"]
    if "NonTrans" in specialized.name():
        args.append("dot")
        if "_Unit_" in specialized.name():
            args.append("xCopy")
    else:
        args.append("xCopy")

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_trmv_row_major_NonTrans_stride_1(
    trmv,
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_trmv(trmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    stride_1 = parallelize_reduction(
        stride_1,
        stride_1.find_loop("j"),
        "dot",
        VEC_W,
        VECTORIZATION_INTERLEAVE_FACTOR,
        memory,
        precision,
    )
    loop_cursor = stride_1.find_loop("jo").body()[0].body()[0]
    stride_1 = vectorize(stride_1, loop_cursor, VEC_W, memory, precision)
    stride_1 = interleave_execution(
        stride_1, stride_1.find_loop("jm"), VECTORIZATION_INTERLEAVE_FACTOR
    )
    stride_1 = simplify(stride_1)
    stride_1 = replace_all(stride_1, instructions)

    return simplify(stride_1)


def schedule_trmv_row_major_NonTrans_Unit_stride_1(
    trmv,
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_trmv(trmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    stride_1 = parallelize_reduction(
        stride_1,
        stride_1.find_loop("j"),
        "dot",
        VEC_W,
        VECTORIZATION_INTERLEAVE_FACTOR,
        memory,
        precision,
    )
    loop_cursor = stride_1.find_loop("jo").body()[0].body()[0]
    stride_1 = vectorize(stride_1, loop_cursor, VEC_W, memory, precision)
    loop_cursor = stride_1.find_loop("jm")
    stride_1 = interleave_execution(
        stride_1, loop_cursor, VECTORIZATION_INTERLEAVE_FACTOR
    )
    stride_1 = simplify(stride_1)
    stride_1 = interleave_outer_loop_with_inner_loop(
        stride_1,
        stride_1.find_loop("i"),
        stride_1.find_loop("jo"),
        ROWS_INTERLEAVE_FACTOR,
    )
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = apply_to_block(stride_1, stride_1.find_loop("ii").body(), hoist_stmt)
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = replace_all(stride_1, instructions)
    stride_1 = set_memory(stride_1, "dot", DRAM_STATIC)

    return simplify(stride_1)


def schedule_trmv_row_major_Trans_Unit_stride_1(
    trmv,
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_trmv(trmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    loop_cursor = stride_1.find_loop("j")
    stride_1 = vectorize(stride_1, loop_cursor, VEC_W, memory, precision)
    stride_1 = interleave_execution(
        stride_1, loop_cursor, VECTORIZATION_INTERLEAVE_FACTOR
    )
    stride_1 = simplify(stride_1)
    stride_1 = interleave_outer_loop_with_inner_loop(
        stride_1,
        stride_1.find_loop("i #1"),
        stride_1.find_loop("joo"),
        ROWS_INTERLEAVE_FACTOR,
    )
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = apply_to_block(stride_1, stride_1.find_loop("ii").body(), hoist_stmt)
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = simplify(stride_1)
    stride_1 = replace_all(stride_1, instructions)
    return simplify(stride_1)


def schedule_trmv_row_major_Trans_stride_1(
    trmv,
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_trmv(trmv, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")

    loop_cursor = stride_1.find_loop("j")
    stride_1 = vectorize(stride_1, loop_cursor, VEC_W, memory, precision)
    stride_1 = interleave_execution(
        stride_1, stride_1.find_loop("jo"), VECTORIZATION_INTERLEAVE_FACTOR
    )
    stride_1 = simplify(stride_1)
    stride_1 = replace_all(stride_1, instructions)
    return simplify(stride_1)


#################################################
# Kernel Parameters
#################################################

ROWS_INTERLEAVE_FACTOR = 4
VECTORIZATION_INTERLEAVE_FACTOR = 2

#################################################
# Generate Entry Points
#################################################

template_sched_list = [
    (
        trmv_row_major_Lower_NonTrans_NonUnit_template,
        schedule_trmv_row_major_NonTrans_stride_1,
    ),
    (
        trmv_row_major_Upper_NonTrans_NonUnit_template,
        schedule_trmv_row_major_NonTrans_stride_1,
    ),
    (
        trmv_row_major_Lower_NonTrans_Unit_template,
        schedule_trmv_row_major_NonTrans_Unit_stride_1,
    ),
    (
        trmv_row_major_Upper_NonTrans_Unit_template,
        schedule_trmv_row_major_NonTrans_Unit_stride_1,
    ),
    (
        trmv_row_major_Lower_Trans_NonUnit_template,
        schedule_trmv_row_major_Trans_stride_1,
    ),
    (
        trmv_row_major_Upper_Trans_NonUnit_template,
        schedule_trmv_row_major_Trans_stride_1,
    ),
    (
        trmv_row_major_Lower_Trans_Unit_template,
        schedule_trmv_row_major_Trans_Unit_stride_1,
    ),
    (
        trmv_row_major_Upper_Trans_Unit_template,
        schedule_trmv_row_major_Trans_Unit_stride_1,
    ),
]

for vec_width, precision in (
    (C.Machine.vec_width, "f32"),
    (C.Machine.vec_width // 2, "f64"),
):
    instructions = [
        C.Machine[f"load_instr_{precision}"],
        C.Machine[f"load_backwards_instr_{precision}"],
        C.Machine[f"store_instr_{precision}"],
        C.Machine[f"store_backwards_instr_{precision}"],
        C.Machine[f"fmadd_instr_{precision}"],
        C.Machine[f"reg_copy_instr_{precision}"],
        C.Machine[f"set_zero_instr_{precision}"],
        C.Machine[f"broadcast_instr_{precision}"],
        C.Machine[f"assoc_reduce_add_instr_{precision}"],
        C.Machine[f"assoc_reduce_add_{precision}_buffer"],
    ]

    for template, sched in template_sched_list:
        proc_stride_any = generate_stride_any_proc(template, specialize_trmv, precision)
        export_exo_proc(globals(), proc_stride_any)
        proc_stride_1 = sched(
            template,
            vec_width,
            VECTORIZATION_INTERLEAVE_FACTOR,
            min(ROWS_INTERLEAVE_FACTOR, vec_width),
            C.Machine.mem_type,
            instructions,
            precision,
        )
        export_exo_proc(globals(), proc_stride_1)

### EXO_LOC SCHEDULE END ###

if __name__ == "__main__":
    pass
