from __future__ import annotations
import os
import sys
from pathlib import Path

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API import compile_procs

from blas_common_schedules import *
import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)


@proc
def gemv_row_major_NonTrans(
    m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][n], y: [R][m]
):
    assert stride(A, 1) == 0

    for i in seq(0, m):
        result: R
        result = 0.0
        for j in seq(0, n):
            result += x[j] * A[i, j]
        y[i] = beta * y[i] + alpha * result


@proc
def gemv_row_major_Trans(
    m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][m], y: [R][n]
):
    for i in seq(0, n):
        y[i] = beta * y[i]
    for i in seq(0, m):
        alphaXi: R
        alphaXi = alpha * x[i]
        for j in seq(0, n):
            y[j] += alphaXi * A[i, j]


def specialize_gemv(gemv, precision):
    prefix = "s" if precision == "f32" else "d"
    name = gemv.name()
    name = name.replace("_template", "")
    specialized = rename(gemv, "exo_" + prefix + name)

    args = ["A", "x", "y", "alpha", "beta"]
    if "NonTrans" in gemv.name():
        args.append("result")
    else:
        args.append("alphaXi")

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_NonTrans(
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_gemv(gemv_row_major_NonTrans, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    # stride_1 = parallelize_reduction(
    #     stride_1,
    #     stride_1.find_loop("j"),
    #     "result",
    #     VEC_W,
    #     VECTORIZATION_INTERLEAVE_FACTOR,
    #     memory,
    #     precision,
    # )
    # loop_cursor = stride_1.find_loop("jo").body()[0].body()[0]
    # stride_1 = vectorize(stride_1, loop_cursor, VEC_W, memory, precision)
    # stride_1 = interleave_execution(
    #     stride_1, stride_1.find_loop("jm"), VECTORIZATION_INTERLEAVE_FACTOR
    # )
    # stride_1 = interleave_outer_loop_with_inner_loop(
    #     stride_1,
    #     stride_1.find_loop("i"),
    #     stride_1.find_loop("jo"),
    #     ROWS_INTERLEAVE_FACTOR,
    # )
    # stride_1 = apply_to_block(
    #     stride_1, stride_1.find_loop("jo").body()[0].body(), hoist_stmt
    # )
    # stride_1 = set_memory(stride_1, "result", DRAM_STATIC)
    # stride_1 = replace_all(stride_1, instructions)
    # stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    # stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    # stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))

    return simplify(stride_1)


def schedule_Trans(
    VEC_W,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    memory,
    instructions,
    precision,
):
    stride_1 = specialize_gemv(gemv_row_major_Trans, precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")

    # stride_1 = vectorize(stride_1, stride_1.find_loop("j"), VEC_W, memory, precision)
    # stride_1 = interleave_execution(
    #     stride_1, stride_1.find_loop("jo"), VECTORIZATION_INTERLEAVE_FACTOR
    # )
    # stride_1 = interleave_outer_loop_with_inner_loop(
    #     stride_1, stride_1.find_loop("i #1"), stride_1.find_loop("joo"), ROWS_INTERLEAVE_FACTOR
    # )
    # stride_1 = apply_to_block(
    #     stride_1, stride_1.find_loop("joo").body()[0].body(), hoist_stmt
    # )
    # stride_1 = hoist_stmt(stride_1, stride_1.find("y[_] #1").parent().parent())
    # stride_1 = replace_all(stride_1, instructions)

    return simplify(stride_1)


#################################################
# Kernel Parameters
#################################################

VECTORIZATION_INTERLEAVE_FACTOR = 2
ROWS_INTERLEAVE_FACTOR = 4

#################################################
# Generate specialized kernels for f32 precision
#################################################

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.set_zero_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.assoc_reduce_add_instr_f32,
    C.Machine.assoc_reduce_add_f32_buffer,
    C.Machine.reg_copy_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

exo_sgemv_row_major_NonTrans_stride_any = specialize_gemv(
    gemv_row_major_NonTrans, "f32"
)
exo_sgemv_row_major_NonTrans_stride_any = rename(
    exo_sgemv_row_major_NonTrans_stride_any,
    exo_sgemv_row_major_NonTrans_stride_any.name() + "_stride_any",
)
exo_sgemv_row_major_Trans_stride_any = specialize_gemv(gemv_row_major_Trans, "f32")
exo_sgemv_row_major_Trans_stride_any = rename(
    exo_sgemv_row_major_Trans_stride_any,
    exo_sgemv_row_major_Trans_stride_any.name() + "_stride_any",
)

exo_sgemv_row_major_NonTrans_stride_1 = schedule_NonTrans(
    C.Machine.vec_width,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)
exo_sgemv_row_major_Trans_stride_1 = schedule_Trans(
    C.Machine.vec_width,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

f64_instructions = [
    C.Machine.load_instr_f64,
    C.Machine.store_instr_f64,
    C.Machine.set_zero_instr_f64,
    C.Machine.fmadd_instr_f64,
    C.Machine.assoc_reduce_add_instr_f64,
    C.Machine.assoc_reduce_add_f64_buffer,
    C.Machine.reg_copy_instr_f64,
    C.Machine.broadcast_instr_f64,
    C.Machine.broadcast_scalar_instr_f64,
]

exo_dgemv_row_major_NonTrans_stride_any = specialize_gemv(
    gemv_row_major_NonTrans, "f64"
)
exo_dgemv_row_major_NonTrans_stride_any = rename(
    exo_dgemv_row_major_NonTrans_stride_any,
    exo_dgemv_row_major_NonTrans_stride_any.name() + "_stride_any",
)
exo_dgemv_row_major_Trans_stride_any = specialize_gemv(gemv_row_major_Trans, "f64")
exo_dgemv_row_major_Trans_stride_any = rename(
    exo_dgemv_row_major_Trans_stride_any,
    exo_dgemv_row_major_Trans_stride_any.name() + "_stride_any",
)

exo_dgemv_row_major_NonTrans_stride_1 = schedule_NonTrans(
    C.Machine.vec_width // 2,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)
exo_dgemv_row_major_Trans_stride_1 = schedule_Trans(
    C.Machine.vec_width // 2,
    VECTORIZATION_INTERLEAVE_FACTOR,
    ROWS_INTERLEAVE_FACTOR,
    C.Machine.mem_type,
    f64_instructions,
    "f64",
)

entry_points = [
    exo_sgemv_row_major_NonTrans_stride_any,
    exo_sgemv_row_major_Trans_stride_any,
    exo_sgemv_row_major_NonTrans_stride_1,
    exo_sgemv_row_major_Trans_stride_1,
    exo_dgemv_row_major_NonTrans_stride_any,
    exo_dgemv_row_major_Trans_stride_any,
    exo_dgemv_row_major_NonTrans_stride_1,
    exo_dgemv_row_major_Trans_stride_1,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
