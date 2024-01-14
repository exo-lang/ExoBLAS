from __future__ import annotations

import math

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

import exo_blas_config as C
from composed_schedules import *
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


def schedule_op_gemm_matmul_no_mem_sys_tiling(
    gemm, k_loop, max_K, max_M, max_N, params
):
    gemm = gemm.add_assertion(f"K <= {max_K}")
    gemm = gemm.add_assertion(f"M <= {max_M}")
    gemm = gemm.add_assertion(f"N <= {max_N}")
    original_gemm = gemm

    k_loop = gemm.find_loop("k")
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
    gemm, _ = tile_loops_top_down(
        gemm, [(i_loop, best_m), (j_loop, params.vec_width * best_n)]
    )
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
        gemm = scalar_to_simd(
            gemm, inner_loop, params.vec_width, params.mem_type, params.precision
        )

    # Hoist A broadcast across (vec_width x n) columns of B
    inner_j_loop = gemm.forward(inner_j_loop)
    gemm = hoist_from_loop(gemm, inner_j_loop)

    gemm = simplify(gemm)

    # Hoisting causes `inner_j_loop` to disappear
    inner_i_loop = gemm.forward(inner_i_loop)
    inner_j_loop = inner_i_loop.body()[2]

    # Don't dynamically index into register arrays
    for loop in (
        C_reg_init_inner_loop,
        C_reg_init_outer_loop,
        inner_j_loop,
    ):
        gemm = unroll_loop(gemm, loop)

    # Interleave accumulate loop, shouldn't exceed ISA registers since
    # compilers will fuse the load from C with the reduction
    gemm = reorder_loops(gemm, C_accum_back_outer_loop)
    gemm = interleave_execution(gemm, C_accum_back_outer_loop, best_m)
    gemm = interleave_execution(gemm, C_accum_back_inner_loop, best_n)

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

    # TODO: Is it always a good idea to copy A and B?
    gemm = ordered_stage_expr(
        gemm, gemm.find("B[_]"), "B_repacked_access_order", params.precision, 5
    )
    B_repacked_access_order = gemm.find("B_repacked_access_order : _")
    gemm = set_memory(gemm, B_repacked_access_order, DRAM_STATIC)
    gemm = bound_alloc(
        gemm,
        B_repacked_access_order,
        [math.ceil(max_N / (best_n * params.vec_width)), max_K, None, None],
        unsafe_disable_checks=True,
    )

    # TODO: we don't want any template pattern matching here
    gemm = scalar_to_simd(
        gemm, gemm.find_loop("i0i"), params.vec_width, params.mem_type, params.precision
    )
    gemm = interleave_execution(gemm, gemm.find_loop("i0o"), best_n)
    gemm = unroll_loop(gemm, gemm.find_loop("i0o"))

    gemm = ordered_stage_expr(
        gemm, gemm.find("A[_]"), "A_repacked_access_order", params.precision, 5
    )
    A_repacked_access_order = gemm.find("A_repacked_access_order : _")
    gemm = set_memory(gemm, A_repacked_access_order, DRAM_STATIC)
    gemm = bound_alloc(
        gemm,
        A_repacked_access_order,
        [math.ceil(max_M / best_m), max_K, None],
        unsafe_disable_checks=True,
    )
    gemm = unroll_loop(gemm, gemm.find_loop("ii"))
    gemm = unroll_loop(gemm, gemm.find_loop("ii"))

    # Instructions...
    gemm = replace_all(gemm, params.instructions)
    gemm = simplify(gemm)

    # TODO: This was found by experimentation, there should be a better way to find why 4
    # is the right answer
    gemm, cursors = auto_divide_loop(gemm, gemm.find_loop("k #2"), 4, tail="cut")
    gemm = unroll_loop(gemm, cursors.inner_loop)

    return original_gemm, simplify(gemm), best_m, best_n * params.vec_width


def schedule_outer_product_gemm_as_tiles(gemm, k_loop, k_tile, i_tile, j_tile, params):
    # Get inner gemm
    (
        inner_gemm_base,
        gemm_no_mem_sys_tiling,
        m,
        n,
    ) = schedule_op_gemm_matmul_no_mem_sys_tiling(
        gemm, k_loop, k_tile, i_tile, j_tile, params
    )
    gemm_no_mem_sys_tiling = rename(
        gemm_no_mem_sys_tiling, f"{gemm_no_mem_sys_tiling.name()}_no_mem_sys_tiling"
    )

    k_loop = gemm.forward(k_loop)
    i_loop = k_loop.body()[0]
    j_loop = i_loop.body()[0]

    tiled_gemm = tile_loops_bottom_up(gemm, k_loop, [k_tile, i_tile, j_tile])

    tiled_gemm = replace_all(tiled_gemm, [inner_gemm_base])
    for i in range(0, 8):
        tiled_gemm = call_eqv(
            tiled_gemm,
            tiled_gemm.find(f"{inner_gemm_base.name()}(_)"),
            gemm_no_mem_sys_tiling,
        )

    return tiled_gemm


def schedule_gemm_matmul(gemm, params):
    gemm = generate_stride_any_proc(gemm, params.precision)

    i_loop = gemm.find_loop("i")
    j_loop = i_loop.body()[0]
    k_loop = j_loop.body()[0]

    # Turn into an outer product
    gemm = reorder_loops(gemm, j_loop)
    gemm = reorder_loops(gemm, i_loop)

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

    CoffeLake_STLB = 1536
    elem_size = 4 if params.precision == "f32" else 8
    k_tile = CoffeLake_STLB // 2 // elem_size
    i_tile = 4096
    j_tile = 3072  # should be 4096, using this for now until inner gemm tail cases are implemented.

    tiled_gemm = schedule_outer_product_gemm_as_tiles(
        gemm, k_loop, k_tile, i_tile, j_tile, params
    )

    return tiled_gemm


template_sched_list = [
    (gemm_matmul_template, schedule_gemm_matmul),
]

for precision in ("f32",):
    for template, sched in template_sched_list:
        proc_stride_1 = sched(
            template,
            Level_3_Params(precision=precision),
        )
        export_exo_proc(globals(), proc_stride_1)
