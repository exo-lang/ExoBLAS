from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

import exo_blas_config as C
from composed_schedules import *
from blas_composed_schedules import blas_vectorize
from codegen_helpers import (
    generate_stride_any_proc,
    export_exo_proc,
)
from parameters import Level_3_Params


@proc
def gemm_matmul_template(M: size, N: size, K: size, A: R[M, K], B: R[K, N], C: R[M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_op_gemm_matmul(gemm, k_loop, params):
    k_loop = gemm.forward(k_loop)
    i_loop = k_loop.body()[0]
    j_loop = i_loop.body()[0]

    # Cast as a (m x K x (vec_width x n) ) matmul
    best_m = 1
    best_n = 1
    registers_used = lambda m, n: m * n + n + 1
    best_registers_used = registers_used(best_m, best_n)
    max_gp_registers = C.Machine.n_vec_registers // 2  # Heuristic
    for m in range(1, max_gp_registers):
        for n in range(1, C.Machine.n_vec_registers):
            guess_registers_used = registers_used(m, n)
            if guess_registers_used > C.Machine.n_vec_registers:
                break
            if guess_registers_used > best_registers_used or (
                guess_registers_used == best_registers_used and m * n > best_m * best_n
            ):
                best_registers_used = guess_registers_used
                best_m = m
                best_n = n

    gemm = tile_loops(gemm, [(i_loop, best_m), (j_loop, params.vec_width * best_n)])

    outer_i_loop = gemm.forward(i_loop)
    outer_j_loop = gemm.forward(j_loop)
    inner_i_loop = outer_j_loop.body()[0]
    inner_j_loop = inner_i_loop.body()[0]

    # Decompose into 3 gemms
    gemm = fission(gemm, outer_i_loop.after())
    gemm = fission(gemm, outer_j_loop.after(), n_lifts=2)

    # Calculate entire C as (m x K x vec_width) using outer product
    gemm = reorder_loops(gemm, k_loop)
    gemm = reorder_loops(gemm, k_loop)

    fma_stmt = inner_j_loop.body()[0]

    # Stage C tile outside the outer product and into accelerator memory
    gemm = simplify(auto_stage_mem(gemm, fma_stmt, "C_reg", n_lifts=3, accum=True))

    k_loop = gemm.forward(k_loop)
    C_reg_alloc = k_loop.prev().prev()
    C_reg_init_outer_loop = k_loop.prev()
    C_reg_init_inner_loop = C_reg_init_outer_loop.body()[0]
    C_accum_back_outer_loop = k_loop.next()
    C_accum_back_inner_loop = C_accum_back_outer_loop.body()[0]

    gemm = set_memory(gemm, C_reg_alloc, params.mem_type)
    gemm = divide_dim(gemm, C_reg_alloc, 1, params.vec_width)

    fma_stmt = gemm.forward(fma_stmt)
    B_read = fma_stmt.rhs().rhs()  # We are assuming B is the rhs

    # Stage B vector to load once across rows
    gemm = simplify(auto_stage_mem(gemm, B_read, "B_reg", n_lifts=2))

    inner_i_loop = gemm.forward(inner_i_loop)
    B_reg_alloc = inner_i_loop.prev().prev()
    B_load_loop = inner_i_loop.prev()

    gemm = set_memory(gemm, B_reg_alloc, params.mem_type)
    gemm = divide_dim(gemm, B_reg_alloc, 0, params.vec_width)

    # Vectorize loops
    for inner_loop in (
        C_reg_init_inner_loop,
        B_load_loop,
        inner_j_loop,
        C_accum_back_inner_loop,
    ):
        gemm = vectorize_to_loops(
            gemm, inner_loop, params.vec_width, params.mem_type, params.precision
        )

    # Hoist A broadcast across (vec_width x n) columns of B
    inner_j_loop = gemm.forward(inner_j_loop)
    gemm = apply_to_block(gemm, inner_j_loop.body(), hoist_stmt)

    gemm = simplify(gemm)

    # Hoisting causes `inner_j_loop` to disappear
    inner_i_loop = gemm.forward(inner_i_loop)
    inner_j_loop = inner_i_loop.body()[2]

    # Don't dynamically index into register arrays
    for loop in (
        C_reg_init_inner_loop,
        C_reg_init_outer_loop,
        B_load_loop,
        inner_j_loop,
        inner_i_loop,
    ):
        gemm = unroll_loop(gemm, loop)

    # Interleave accumulate loop, shouldn't exceed ISA registers since
    # compilers will fuse the load from C with the reduction
    gemm = interleave_execution(gemm, C_accum_back_inner_loop, best_n)
    gemm = interleave_execution(gemm, C_accum_back_outer_loop, best_m)

    # Instructions...
    gemm = replace_all(gemm, params.instructions)

    return simplify(gemm)


def schedule_gemm_matmul(gemm, params):
    gemm = generate_stride_any_proc(gemm, params.precision)

    i_loop = gemm.find_loop("i")
    j_loop = i_loop.body()[0]
    k_loop = j_loop.body()[0]

    # Turn into an outer product
    gemm = reorder_loops(gemm, j_loop)
    gemm = reorder_loops(gemm, i_loop)

    return schedule_op_gemm_matmul(gemm, k_loop, params)


template_sched_list = [
    (gemm_matmul_template, schedule_gemm_matmul),
]

for precision in ("f32", "f64"):
    for template, sched in template_sched_list:
        proc_stride_1 = sched(
            template,
            Level_3_Params(precision=precision),
        )
        export_exo_proc(globals(), proc_stride_1)
