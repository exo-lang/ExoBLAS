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
    vectorize_to_loops,
    interleave_execution,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
    vectorize,
)


### EXO_LOC ALGORITHM START ###
@proc
def gemv_row_major_NonTrans(
    m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][n], y: [R][m]
):
    assert stride(A, 1) == 1

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
    assert stride(A, 1) == 1

    for i in seq(0, n):
        y[i] = beta * y[i]
    for i in seq(0, m):
        alphaXi: R
        alphaXi = alpha * x[i]
        for j in seq(0, n):
            y[j] += alphaXi * A[i, j]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
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
    return stride_1
    stride_1 = vectorize(
        stride_1,
        stride_1.find_loop("j"),
        VEC_W,
        VECTORIZATION_INTERLEAVE_FACTOR,
        VECTORIZATION_INTERLEAVE_FACTOR,
        memory,
        precision,
        [],
        vectorize_tail=memory == AVX2,
    )

    stride_1 = interleave_outer_loop_with_inner_loop(
        stride_1,
        stride_1.find_loop("i"),
        stride_1.find_loop("joo"),
        ROWS_INTERLEAVE_FACTOR,
    )
    stride_1 = apply_to_block(
        stride_1, stride_1.find_loop("joo").body()[0].body(), hoist_stmt
    )
    stride_1 = set_memory(stride_1, "result", DRAM_STATIC)
    stride_1 = replace_all(stride_1, instructions)
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("joi"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("joi"))
    stride_1 = stage_mem(stride_1, stride_1.body(), "alpha", "alpha_")
    stride_1 = stage_mem(stride_1, stride_1.body(), "beta", "beta_")
    stride_1 = simplify(stride_1)

    return stride_1


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

    stride_1 = stage_mem(stride_1, stride_1.body(), "alpha", "alpha_")
    stride_1 = stage_mem(stride_1, stride_1.body(), "beta", "beta_")
    stride_1 = vectorize_to_loops(
        stride_1, stride_1.find_loop("j"), VEC_W, memory, precision
    )
    stride_1 = interleave_execution(
        stride_1, stride_1.find_loop("jo"), VECTORIZATION_INTERLEAVE_FACTOR
    )
    stride_1 = interleave_outer_loop_with_inner_loop(
        stride_1,
        stride_1.find_loop("i #1"),
        stride_1.find_loop("joo"),
        ROWS_INTERLEAVE_FACTOR,
    )
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = apply_to_block(
        stride_1, stride_1.find_loop("joo").body()[0].body(), hoist_stmt
    )
    window_expr = (
        lambda offset, ji: f"{VEC_W} * ({VECTORIZATION_INTERLEAVE_FACTOR} * joo + {offset}) + {ji}"
    )
    stride_1 = stage_mem(
        stride_1,
        stride_1.find_loop("ii"),
        f"y[{window_expr(0, 0)}:{window_expr(VECTORIZATION_INTERLEAVE_FACTOR - 1, VEC_W)}]",
        f"yReg",
    )
    stride_1 = simplify(stride_1)
    stride_1 = divide_dim(stride_1, "yReg", 0, VEC_W)
    stride_1 = set_memory(stride_1, "yReg", memory)
    for i in range(VECTORIZATION_INTERLEAVE_FACTOR - 1):
        stride_1 = cut_loop(stride_1, f"for i0 in _:_ #{i}", VEC_W)
        stride_1 = shift_loop(stride_1, f"for i0 in _:_ #{i + 1}", 0)
    for i in range(
        VECTORIZATION_INTERLEAVE_FACTOR, 2 * VECTORIZATION_INTERLEAVE_FACTOR - 1
    ):
        stride_1 = cut_loop(stride_1, f"for i0 in _:_ #{i}", VEC_W)
        stride_1 = shift_loop(stride_1, f"for i0 in _:_ #{i + 1}", 0)
    stride_1 = simplify(stride_1)
    stride_1 = replace_all(stride_1, instructions)
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = fission(stride_1, stride_1.find_loop("joi").after())
    stride_1 = reorder_loops(stride_1, stride_1.find_loop("ii"))
    stride_1 = reorder_loops(stride_1, stride_1.find_loop("ii #1"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = unroll_loop(stride_1, stride_1.find_loop("ii"))
    stride_1 = set_memory(stride_1, "alphaXi", DRAM_STATIC)
    return simplify(stride_1)


#################################################
# Kernel Parameters
#################################################

VECTORIZATION_INTERLEAVE_FACTOR = 2
ROWS_INTERLEAVE_FACTOR = 4

#################################################
# Generate specialized kernels for f32 precision
#################################################

f32_instructions = C.Machine.get_instructions("f32")

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
    ROWS_INTERLEAVE_FACTOR * 2,
    C.Machine.mem_type,
    f32_instructions,
    "f32",
)

#################################################
# Generate specialized kernels for f64 precision
#################################################

f64_instructions = C.Machine.get_instructions("f64")

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
### EXO_LOC SCHEDULE END ###

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
