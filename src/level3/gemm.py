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


def specialize_micro(gemm_uk, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    _, init, main_k, axpy = gemm_uk.body()

    main_i = get_inner_loop(gemm_uk, main_k)
    j_loops = [get_inner_loop(gemm_uk, c) for c in (init, main_i, axpy)]

    gemm_uk = simplify(apply(lambda p, c: round_loop(p, c, vw))(gemm_uk, j_loops))
    j_loop = gemm_uk.forward(j_loops[0])

    specialize_i = not is_loop_bounds_const(gemm_uk, main_i)
    specialize_j = not is_loop_bounds_const(gemm_uk, j_loop)
    make_conds = lambda e, mx: [f"{expr_to_string(e)} == {i + 1}" for i in range(mx)]

    if specialize_i:
        gemm_uk = specialize(gemm_uk, gemm_uk.body(), make_conds(main_i.hi(), m_r))
    if specialize_j:
        for tile in gemm_uk.find("C_tile:_", many=True):
            gemm_uk = specialize(
                gemm_uk, tile.expand(), make_conds(j_loop.hi().lhs(), n_r_fac)
            )
    gemm_uk = dce(simplify(gemm_uk))
    return gemm_uk


def schedule_micro(gemm_uk, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    gemm_uk = specialize_micro(gemm_uk, precision, machine, m_r, n_r_fac)

    def rewrite(gemm_uk):
        tile, init, main_k, axpy = gemm_uk.body()
        main_i = get_inner_loop(gemm_uk, main_k)
        main_j = get_inner_loop(gemm_uk, main_i)
        m_r = main_i.hi().value()
        n_r = main_j.hi().value()

        gemm_uk = rename(gemm_uk, f"{gemm_uk.name()}_{m_r}x{n_r}")
        gemm_uk = set_memory(gemm_uk, tile, machine.mem_type)
        gemm_uk = divide_dim(gemm_uk, tile, 1, vw)

        loops = [init, main_i, axpy]

        return gemm_uk

    blocks = map(lambda c: c.expand(), gemm_uk.find("C_tile:_", many=True))
    gemm_uk = apply(extract_and_schedule(rewrite))(gemm_uk, blocks, gemm_uk.name())
    return gemm_uk


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
    gemm_compute = repeate_n(lift_scope)(gemm_compute, gemm_compute.find_loop("k"), n=2)
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
    gemm_mk, _ = extract_subproc(gemm_mk, cursors.load, gemm_mk.name() + "A_pack")

    packed_B_shape = ((1, max_N // n_r), (0, max_K), (1, n_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "B", packed_B_shape, "packed_B", rc=1)
    # TODO: Schedule packing kernel
    gemm_mk, _ = extract_subproc(gemm_mk, cursors.load, gemm_mk.name() + "B_pack")

    gemm_mk = extract_and_schedule(schedule_compute)(
        gemm_mk, i_loop, gemm_mk.name() + "_compute", precision, machine, m_r, n_r_fac
    )
    return gemm_mk_starter, gemm_mk


def schedule(main_gemm, i_loop, precision, machine):
    m_r = 4
    n_r_fac = 3
    vw = machine.vec_width(precision)

    M_tile = m_r * 512
    N_tile = n_r_fac * vw * 512
    K_tile = 512

    gemm_macro = schedule_macro(
        gemm, precision, machine, M_tile, N_tile, K_tile, m_r, n_r_fac
    )

    gemm_tiled = reorder_loops(
        main_gemm, i_loop.body()[0]
    )  # Iterate over the main gemm as (i, k, j)
    gemm_tiled = tile_loops_bottom_up(gemm_tiled, i_loop, [M_tile, K_tile, N_tile])
    gemm_tiled = apply(reorder_loops)(
        gemm_tiled, gemm_tiled.find_loop("ki", many=True)
    )  # Change macrokernel loops to original order

    gemm_tiled = replace_all_stmts(gemm_tiled, [gemm_macro])
    macro_calls = filter_cursors(is_call)(gemm_tiled, nlr_stmts(gemm_tiled))
    gemm_tiled = simplify(apply(inline_proc_and_wins)(gemm_tiled, macro_calls))

    gemm_tiled = apply(hoist_from_loop)(
        gemm_tiled, gemm_tiled.find_loop("jo", many=True)
    )
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_A : _", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_B : _", many=True))
    return gemm_tiled


variants_generator(schedule, ("f32",))(gemm, "i", globals=globals())
