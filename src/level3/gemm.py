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
def gemm_matmul_template(
    M: size, N: size, K: size, A: [R][M, K], B: [R][K, N], C: [R][M, N]
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


def schedule_op_gemm_matmul_no_mem_sys_tiling(gemm, k_loop, params):
    k_loop = gemm.forward(k_loop)
    i_loop = k_loop.body()[0]
    j_loop = i_loop.body()[0]

    # Cast as a (m x K x (vec_width x n) ) matmul

    # Solve the constraints for m and n
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
    gemm = simplify(gemm)

    outer_i_loop = gemm.forward(outer_i_loop)
    A_times_B_strip_gemm_k_loop = outer_i_loop.next()
    A_strip_times_B_gemm_k_loop = outer_i_loop.next().next()

    # Turn A times B strip into a 3 loop outer product gemm
    A_times_B_strip_gemm_outer_i_loop = A_times_B_strip_gemm_k_loop.body()[0]
    gemm = mult_loops(
        gemm,
        A_times_B_strip_gemm_outer_i_loop,
        A_times_B_strip_gemm_outer_i_loop.name()[:-1],
    )

    return simplify(gemm), best_m, best_n * params.vec_width


def schedule_outer_product_gemm_as_tiles(
    gemm, k_loop, k_tile, i_tile, j_tile, gemm_no_mem_sys_tiling, params
):
    untiled_gemm = gemm

    k_loop = gemm.forward(k_loop)
    i_loop = k_loop.body()[0]
    j_loop = i_loop.body()[0]

    # Tile problem
    tiled_gemm = tile_loops(
        gemm, [(k_loop, k_tile), (i_loop, i_tile), (j_loop, j_tile)]
    )

    # Copy A
    tiled_gemm = simplify(
        auto_stage_mem(tiled_gemm, tiled_gemm.find("A[_]"), "A_copy", n_lifts=4)
    )

    # Vectorize A copy
    i_loop = tiled_gemm.forward(i_loop)
    copy_outer_loop = i_loop.body()[1]
    copy_inner_loop = copy_outer_loop.body()[0]
    tiled_gemm = vectorize_to_loops(
        tiled_gemm, copy_inner_loop, params.vec_width, params.mem_type, params.precision
    )
    interleave_1 = min(C.Machine.n_vec_registers, k_tile // params.vec_width)
    interleave_2 = min(
        i_tile, C.Machine.n_vec_registers // (k_tile // params.vec_width)
    )
    tiled_gemm = interleave_execution(tiled_gemm, copy_inner_loop, interleave_1)
    if interleave_2 > 1:
        tiled_gemm = simplify(tiled_gemm)
        tiled_gemm = unroll_loop(tiled_gemm, copy_inner_loop)
        tiled_gemm = interleave_execution(tiled_gemm, copy_outer_loop, interleave_2)

    tiled_gemm = lift_alloc(tiled_gemm, "A_copy", n_lifts=2)
    tiled_gemm = set_memory(tiled_gemm, "A_copy", DRAM_STATIC)

    tiled_gemm = replace_all(tiled_gemm, [untiled_gemm])
    for i in range(0, 4):
        tiled_gemm = call_eqv(
            tiled_gemm,
            tiled_gemm.find(f"{untiled_gemm.name()}(_)"),
            gemm_no_mem_sys_tiling,
        )

    tiled_gemm = replace_all(tiled_gemm, params.instructions)

    return simplify(tiled_gemm)


def schedule_gemm_matmul(gemm, params):
    gemm = generate_stride_any_proc(gemm, params.precision)

    i_loop = gemm.find_loop("i")
    j_loop = i_loop.body()[0]
    k_loop = j_loop.body()[0]

    # Turn into an outer product
    gemm = reorder_loops(gemm, j_loop)
    gemm = reorder_loops(gemm, i_loop)

    # Get inner gemm
    gemm_no_mem_sys_tiling, m, n = schedule_op_gemm_matmul_no_mem_sys_tiling(
        gemm, k_loop, params
    )
    gemm_no_mem_sys_tiling = rename(
        gemm_no_mem_sys_tiling, f"{gemm_no_mem_sys_tiling.name()}_no_mem_sys_tiling"
    )

    # A tile will be a i_tile x k_tile
    # B tile will be a k_tile x j_tile

    # We iterate down B in the inner gemm
    # So, we need to always have space for k_tile pages

    # Once we touch k_tiles pages from B, we need to keep
    # working on the same pages i.e. j_tiles = 4096

    # Not sure if there is a reason for what to choose as
    # i_tiles. If we set, it as i_tiles=j_tiles=4096, then
    # it would be simple to think about the problem because
    # pages will be split evenly between A and B

    k_tile = 80
    i_tile = 4096
    j_tile = 4096

    tiled_gemm = schedule_outer_product_gemm_as_tiles(
        gemm, k_loop, k_tile, i_tile, j_tile, gemm_no_mem_sys_tiling, params
    )

    return tiled_gemm


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
