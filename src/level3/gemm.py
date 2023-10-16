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


def schedule_gemm_matmul(gemm, params):
    gemm = generate_stride_any_proc(gemm, params.precision)

    i_loop = gemm.find_loop("i")
    j_loop = i_loop.body()[0]
    k_loop = j_loop.body()[0]

    # Turn into an outer product
    gemm = reorder_loops(gemm, j_loop)
    gemm = reorder_loops(gemm, i_loop)

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

    # Decompose into 3 gemms
    gemm = fission(gemm, gemm.find_loop("io").after())
    gemm = fission(gemm, gemm.find_loop("jo").after(), n_lifts=2)

    # Calculate entire C as (m x K x vec_width) using outer product
    gemm = reorder_loops(gemm, k_loop)
    gemm = reorder_loops(gemm, k_loop)

    # Stage C tile outside the outer product and into accelerator memory
    gemm = simplify(
        auto_stage_mem(gemm, gemm.find("C[_] += _"), "C_reg", n_lifts=3, accum=True)
    )
    gemm = set_memory(gemm, "C_reg", params.mem_type)
    gemm = divide_dim(gemm, "C_reg", 1, params.vec_width)

    # Stage B vector to load once across rows
    gemm = simplify(auto_stage_mem(gemm, gemm.find("B[_]"), "B_reg", n_lifts=2))
    gemm = set_memory(gemm, "B_reg", params.mem_type)
    gemm = divide_dim(gemm, "B_reg", 0, params.vec_width)

    # Vectorize loops
    for iter_name in ("i1", "i0 #1", "ji", "i1"):
        inner_loop_cursor = gemm.find_loop(iter_name)
        gemm = vectorize_to_loops(
            gemm, inner_loop_cursor, params.vec_width, params.mem_type, params.precision
        )

    # Hoist A broadcast across (vec_width x n) columns of B
    gemm = apply_to_block(gemm, gemm.find_loop("jio").body(), hoist_stmt)

    gemm = simplify(gemm)
    gemm = replace_all(gemm, params.instructions)

    # Don't dynamically index into register arrays
    for loop in ("i1o", "i0", "i0o", "jio", "ii"):
        gemm = unroll_loop(gemm, loop)

    gemm = interleave_execution(gemm, gemm.find_loop("i1o"), best_n)
    gemm = interleave_execution(gemm, gemm.find_loop("i0"), best_m)

    return simplify(gemm)


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
