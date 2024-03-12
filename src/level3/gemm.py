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
        gemm_uk = apply(optimize_level_2)(
            gemm_uk, loops, precision, machine, m_r, n_r // vw, vec_tail="perfect"
        )
        return gemm_uk

    blocks = map(lambda c: c.expand(), gemm_uk.find("C_tile:_", many=True))
    gemm_uk = apply(extract_and_schedule(rewrite))(gemm_uk, blocks, gemm_uk.name())
    return gemm_uk


def schedule_macro(gemm, i_loop, precision, machine, m_r, n_r_fac, do_br=False):
    n_r = machine.vec_width(precision) * n_r_fac

    j_loop = get_inner_loop(gemm, i_loop)
    k_loop = get_inner_loop(gemm, j_loop)

    gemm = auto_stage_mem(gemm, k_loop, "C", "C_tile", accum=True)
    gemm = lift_reduce_constant(gemm, gemm.forward(k_loop).expand(1, 0))
    gemm = inline_assign(gemm, gemm.find("C_tile = _ * _"))

    gemm = tile_loops_bottom_up(gemm, i_loop, (m_r, n_r, None))
    gemm = apply(repeate_n(lift_scope))(gemm, gemm.find_loop("k", many=True), n=2)

    tiles = gemm.find("C_tile:_", many=True)
    names = ["_uk", "_r_uk", "_b_uk", "_br_uk"]
    names = [gemm.name() + su for su in names]
    if not do_br:
        tiles = tiles[:-1]
        names = names[:-1]
    for tile, name in zip(tiles, names):
        gemm = extract_and_schedule(schedule_micro)(
            gemm, tile.expand(), name, precision, machine, m_r, n_r_fac
        )

    gemm = cleanup(gemm)
    print(gemm)
    return gemm


def schedule(gemm, i_loop, precision, machine):
    macro = schedule_macro(gemm, i_loop, precision, machine, 4, 3)
    return macro


variants_generator(schedule, ("f32",))(gemm, "i", globals=globals())
