from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *

import exo_blas_config as C
from stdlib import *
from codegen_helpers import *
from blaslib import *


@proc
def gemm(M: size, N: size, K: size, alpha: R, A: [R][M, K], B: [R][K, N], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * B[k, j])


def schedule_compute(gemm_compute, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac
    i_loop = gemm_compute.body()[0]
    j_loop = get_inner_loop(gemm_compute, i_loop)
    k_loop = get_inner_loop(gemm_compute, j_loop)

    gemm_compute, cs = auto_stage_mem(
        gemm_compute, k_loop, "C", "C_tile", accum=True, rc=1
    )
    gemm_compute = lift_reduce_constant(gemm_compute, cs.load.expand(0, 1))
    assign = gemm_compute.forward(cs.store).prev()
    gemm_compute = inline_assign(gemm_compute, assign)
    gemm_compute = set_memory(gemm_compute, cs.alloc, machine.mem_type)

    gemm_compute = tile_loops_bottom_up(
        gemm_compute, i_loop, (m_r, n_r, None), tail="guard"
    )
    gemm_compute = repeate_n(lift_scope)(gemm_compute, k_loop, n=2)
    gemm_compute = divide_dim(gemm_compute, cs.alloc, 1, vw)

    loops = gemm_compute.find_loop("ii", many=True)
    gemm_compute = apply(optimize_level_2)(
        gemm_compute, loops, precision, machine, m_r, n_r_fac, vec_tail="perfect"
    )

    def cut(proc, loop, cond, rng):
        loop = proc.forward(loop)
        cut_val = FormattedExprStr(f"_ - 1", loop.hi())
        proc, (loop1, loop2) = cut_loop_(proc, loop, cut_val, rc=True)
        proc = specialize(proc, loop2.body(), [f"{cond(loop2, i)} == {i}" for i in rng])
        return proc

    right_cond = lambda l, i: f"(N - {l.name()} * {n_r} + {vw - 1}) / {vw}"
    gemm_compute = cut(gemm_compute, j_loop, right_cond, range(1, n_r_fac))
    bottom_cond = lambda l, i: f"M - {l.name()} * {m_r}"
    gemm_compute = cut(gemm_compute, i_loop, bottom_cond, range(m_r, 1, -1))

    def rewrite(p):
        p = dce(p)
        p = replace_all_stmts(p, machine.get_instructions(precision))
        p = simplify(p)
        try:
            p = delete_pass(p)
        except:
            pass
        return p

    blocks = gemm_compute.find_loop("C_tile:_", many=True)
    for i, tile in enumerate(blocks[:8]):
        name = gemm_compute.name() + str(i)
        gemm_compute = extract_and_schedule(rewrite)(gemm_compute, tile.expand(), name)
    # for i, tile in enumerate(blocks[8:]):
    #     name = gemm_compute.name() + str(8 + i)
    #     gemm_compute = extract_and_schedule(lambda p:p)(gemm_compute, tile.expand(), name)
    return gemm_compute


def schedule_macro(gemm_mk, precision, machine, max_M, max_N, max_K, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac
    gemm_mk = specialize_precision(gemm_mk, precision)
    for var, max_var in zip(("M", "N", "K"), (max_M, max_N, max_K)):
        gemm_mk = gemm_mk.add_assertion(f"{var} <= {max_var}")

    gemm_mk_starter = gemm_mk
    gemm_mk = rename(gemm_mk, gemm_mk.name() + "_mk")
    i_loop = gemm_mk.body()[0]

    packed_A_shape = ((0, max_M // m_r), (1, max_K), (0, m_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "A", packed_A_shape, "packed_A", rc=1)
    # TODO: Schedule packing kernel
    gemm_mk, _ = extract_subproc(gemm_mk, cursors.load, gemm_mk.name() + "_A_pack")

    packed_B_shape = ((1, max_N // n_r), (0, max_K), (1, n_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "B", packed_B_shape, "packed_B", rc=1)
    # TODO: Schedule packing kernel
    gemm_mk, _ = extract_subproc(gemm_mk, cursors.load, gemm_mk.name() + "_B_pack")
    gemm_mk = extract_and_schedule(schedule_compute)(
        gemm_mk, i_loop, gemm_mk.name() + "_compute", precision, machine, m_r, n_r_fac
    )
    return gemm_mk_starter, gemm_mk


def schedule(
    main_gemm, i_loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile
):
    gemm_macro = schedule_macro(
        gemm, precision, machine, M_tile, N_tile, K_tile, m_r, n_r_fac
    )

    # Iterate over the main gemm as (i, k, j)
    gemm_tiled = reorder_loops(main_gemm, i_loop.body()[0])
    gemm_tiled = tile_loops_bottom_up(gemm_tiled, i_loop, [M_tile, K_tile, N_tile])
    gemm_tiled = apply(reorder_loops)(gemm_tiled, gemm_tiled.find_loop("ki", many=True))
    gemm_tiled = replace_all_stmts(gemm_tiled, [gemm_macro])

    macro_calls = filter_cursors(is_call)(gemm_tiled, nlr_stmts(gemm_tiled))
    gemm_tiled = simplify(apply(inline_proc_and_wins)(gemm_tiled, macro_calls))

    gemm_tiled = apply(hoist_from_loop)(
        gemm_tiled, gemm_tiled.find_loop("jo", many=True)
    )
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_A : _", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_B : _", many=True))
    return gemm_tiled


variants_generator(schedule, ("f32",))(
    gemm, "i", 4, 3, 512 * 4, 512 * 24, 512, globals=globals()
)
