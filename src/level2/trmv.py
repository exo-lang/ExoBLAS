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
    trmv, VEC_W, VECTORIZATION_INTERLEAVE_FACTOR, memory, instructions, precision
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

    stride_1 = replace_all(stride_1, instructions)

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
    trmv, VEC_W, VECTORIZATION_INTERLEAVE_FACTOR, memory, instructions, precision
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
# Generate specialized kernels for f32 precision
#################################################

exo_strmv_row_major_Upper_NonTrans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Upper_NonTrans_Unit_template, "f32"
)
exo_strmv_row_major_Upper_NonTrans_Unit_stride_any = rename(
    exo_strmv_row_major_Upper_NonTrans_Unit_stride_any,
    exo_strmv_row_major_Upper_NonTrans_Unit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Upper_NonTrans_NonUnit_template, "f32"
)
exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any = rename(
    exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any,
    exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Lower_NonTrans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Lower_NonTrans_Unit_template, "f32"
)
exo_strmv_row_major_Lower_NonTrans_Unit_stride_any = rename(
    exo_strmv_row_major_Lower_NonTrans_Unit_stride_any,
    exo_strmv_row_major_Lower_NonTrans_Unit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Lower_NonTrans_NonUnit_template, "f32"
)
exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any = rename(
    exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any,
    exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Upper_Trans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Upper_Trans_Unit_template, "f32"
)
exo_strmv_row_major_Upper_Trans_Unit_stride_any = rename(
    exo_strmv_row_major_Upper_Trans_Unit_stride_any,
    exo_strmv_row_major_Upper_Trans_Unit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Upper_Trans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Upper_Trans_NonUnit_template, "f32"
)
exo_strmv_row_major_Upper_Trans_NonUnit_stride_any = rename(
    exo_strmv_row_major_Upper_Trans_NonUnit_stride_any,
    exo_strmv_row_major_Upper_Trans_NonUnit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Lower_Trans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Lower_Trans_Unit_template, "f32"
)
exo_strmv_row_major_Lower_Trans_Unit_stride_any = rename(
    exo_strmv_row_major_Lower_Trans_Unit_stride_any,
    exo_strmv_row_major_Lower_Trans_Unit_stride_any.name() + "_stride_any",
)
exo_strmv_row_major_Lower_Trans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Lower_Trans_NonUnit_template, "f32"
)
exo_strmv_row_major_Lower_Trans_NonUnit_stride_any = rename(
    exo_strmv_row_major_Lower_Trans_NonUnit_stride_any,
    exo_strmv_row_major_Lower_Trans_NonUnit_stride_any.name() + "_stride_any",
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.load_backwards_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.store_backwards_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.reg_copy_instr_f32,
    C.Machine.set_zero_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.assoc_reduce_add_instr_f32,
    C.Machine.assoc_reduce_add_f32_buffer,
]

exo_strmv_row_major_Upper_NonTrans_Unit_stride_1 = (
    schedule_trmv_row_major_NonTrans_Unit_stride_1(
        trmv_row_major_Upper_NonTrans_Unit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
print(exo_strmv_row_major_Upper_NonTrans_Unit_stride_1)
exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_1 = (
    schedule_trmv_row_major_NonTrans_stride_1(
        trmv_row_major_Upper_NonTrans_NonUnit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Lower_NonTrans_Unit_stride_1 = (
    schedule_trmv_row_major_NonTrans_Unit_stride_1(
        trmv_row_major_Lower_NonTrans_Unit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_1 = (
    schedule_trmv_row_major_NonTrans_stride_1(
        trmv_row_major_Lower_NonTrans_NonUnit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Upper_Trans_Unit_stride_1 = (
    schedule_trmv_row_major_Trans_Unit_stride_1(
        trmv_row_major_Upper_Trans_Unit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Lower_Trans_Unit_stride_1 = (
    schedule_trmv_row_major_Trans_Unit_stride_1(
        trmv_row_major_Lower_Trans_Unit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Upper_Trans_NonUnit_stride_1 = (
    schedule_trmv_row_major_Trans_stride_1(
        trmv_row_major_Upper_Trans_NonUnit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)
exo_strmv_row_major_Lower_Trans_NonUnit_stride_1 = (
    schedule_trmv_row_major_Trans_stride_1(
        trmv_row_major_Lower_Trans_NonUnit_template,
        C.Machine.vec_width,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f32_instructions,
        "f32",
    )
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Upper_NonTrans_Unit_template, "f64"
)
exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any = rename(
    exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any,
    exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Upper_NonTrans_NonUnit_template, "f64"
)
exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any = rename(
    exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any,
    exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Lower_NonTrans_Unit_template, "f64"
)
exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any = rename(
    exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any,
    exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Lower_NonTrans_NonUnit_template, "f64"
)
exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any = rename(
    exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any,
    exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Upper_Trans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Upper_Trans_Unit_template, "f64"
)
exo_dtrmv_row_major_Upper_Trans_Unit_stride_any = rename(
    exo_dtrmv_row_major_Upper_Trans_Unit_stride_any,
    exo_dtrmv_row_major_Upper_Trans_Unit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Upper_Trans_NonUnit_template, "f64"
)
exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any = rename(
    exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any,
    exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Lower_Trans_Unit_stride_any = specialize_trmv(
    trmv_row_major_Lower_Trans_Unit_template, "f64"
)
exo_dtrmv_row_major_Lower_Trans_Unit_stride_any = rename(
    exo_dtrmv_row_major_Lower_Trans_Unit_stride_any,
    exo_dtrmv_row_major_Lower_Trans_Unit_stride_any.name() + "_stride_any",
)
exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any = specialize_trmv(
    trmv_row_major_Lower_Trans_NonUnit_template, "f64"
)
exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any = rename(
    exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any,
    exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any.name() + "_stride_any",
)

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.load_backwards_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.store_backwards_instr_f64,
    C.Machine.fmadd_instr_f64,
    C.Machine.reg_copy_instr_f64,
    C.Machine.set_zero_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.assoc_reduce_add_instr_f64,
    C.Machine.assoc_reduce_add_f64_buffer,
]

exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_1 = (
    schedule_trmv_row_major_NonTrans_Unit_stride_1(
        trmv_row_major_Upper_NonTrans_Unit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_1 = (
    schedule_trmv_row_major_NonTrans_stride_1(
        trmv_row_major_Upper_NonTrans_NonUnit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_1 = (
    schedule_trmv_row_major_NonTrans_Unit_stride_1(
        trmv_row_major_Lower_NonTrans_Unit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_1 = (
    schedule_trmv_row_major_NonTrans_stride_1(
        trmv_row_major_Lower_NonTrans_NonUnit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Upper_Trans_Unit_stride_1 = (
    schedule_trmv_row_major_Trans_Unit_stride_1(
        trmv_row_major_Upper_Trans_Unit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Lower_Trans_Unit_stride_1 = (
    schedule_trmv_row_major_Trans_Unit_stride_1(
        trmv_row_major_Lower_Trans_Unit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        ROWS_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_1 = (
    schedule_trmv_row_major_Trans_stride_1(
        trmv_row_major_Upper_Trans_NonUnit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_1 = (
    schedule_trmv_row_major_Trans_stride_1(
        trmv_row_major_Lower_Trans_NonUnit_template,
        C.Machine.vec_width // 2,
        VECTORIZATION_INTERLEAVE_FACTOR,
        C.Machine.mem_type,
        f64_instructions,
        "f64",
    )
)
### EXO_LOC SCHEDULE END ###


entry_points = [
    exo_strmv_row_major_Upper_NonTrans_Unit_stride_any,
    exo_strmv_row_major_Upper_NonTrans_Unit_stride_1,
    exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any,
    exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_1,
    exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_any,
    exo_dtrmv_row_major_Upper_NonTrans_Unit_stride_1,
    exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_any,
    exo_dtrmv_row_major_Upper_NonTrans_NonUnit_stride_1,
    exo_strmv_row_major_Lower_NonTrans_Unit_stride_any,
    exo_strmv_row_major_Lower_NonTrans_Unit_stride_1,
    exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any,
    exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_1,
    exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_any,
    exo_dtrmv_row_major_Lower_NonTrans_Unit_stride_1,
    exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_any,
    exo_dtrmv_row_major_Lower_NonTrans_NonUnit_stride_1,
    exo_strmv_row_major_Upper_Trans_Unit_stride_any,
    exo_strmv_row_major_Upper_Trans_Unit_stride_1,
    exo_strmv_row_major_Upper_Trans_NonUnit_stride_any,
    exo_strmv_row_major_Upper_Trans_NonUnit_stride_1,
    exo_dtrmv_row_major_Upper_Trans_Unit_stride_any,
    exo_dtrmv_row_major_Upper_Trans_Unit_stride_1,
    exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_any,
    exo_dtrmv_row_major_Upper_Trans_NonUnit_stride_1,
    exo_strmv_row_major_Lower_Trans_Unit_stride_any,
    exo_strmv_row_major_Lower_Trans_Unit_stride_1,
    exo_strmv_row_major_Lower_Trans_NonUnit_stride_any,
    exo_strmv_row_major_Lower_Trans_NonUnit_stride_1,
    exo_dtrmv_row_major_Lower_Trans_Unit_stride_any,
    exo_dtrmv_row_major_Lower_Trans_Unit_stride_1,
    exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_any,
    exo_dtrmv_row_major_Lower_Trans_NonUnit_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
