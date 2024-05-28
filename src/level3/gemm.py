from __future__ import annotations
from math import ceil, floor

from exo import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *
from exo.libs.memories import DRAM_STATIC

import exo_blas_config as C
from stdlib import *
from codegen_helpers import *
from blaslib import *
from cblas_enums import *
from memories import *


@proc
def gemm(
    TransA: size,
    TransB: size,
    M: size,
    N: size,
    K: size,
    alpha: R,
    A: [R][M, K],
    B: [R][K, N],
    AT: [R][K, M],
    BT: [R][N, K],
    C: [R][M, N],
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                if TransA == CblasNoTransValue:
                    if TransB == CblasNoTransValue:
                        C[i, j] += alpha * (A[i, k] * B[k, j])
                    else:
                        C[i, j] += alpha * (A[i, k] * BT[j, k])
                else:
                    if TransB == CblasNoTransValue:
                        C[i, j] += alpha * (AT[k, i] * B[k, j])
                    else:
                        C[i, j] += alpha * (AT[k, i] * BT[j, k])


def schedule_macro(gemm_mk, precision, machine, max_M, max_N, max_K, m_r, n_r_fac, TransA, TransB):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac
    gemm_mk = specialize_precision(gemm_mk, precision)
    for var, max_var in zip(("M", "N", "K"), (max_M, max_N, max_K)):
        gemm_mk = gemm_mk.add_assertion(f"{var} <= {max_var}")

    gemm_mk_starter = gemm_mk
    gemm_mk = rename(gemm_mk, gemm_mk.name() + "_mk")
    i_loop = gemm_mk.body()[0]

    isTransA = TransA == CblasTransValue
    packed_A_shape = ((int(isTransA), ceil(max_M / m_r)), (1 - int(isTransA), max_K), (int(isTransA), m_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "AT" if isTransA else "A", packed_A_shape, "packed_A", rc=1)
    gemm_mk = set_memory(gemm_mk, cursors.alloc, ALIGNED_DRAM_STATIC)

    expr = gemm_mk.find(f"(_ + M / {m_r} * {m_r}) / {m_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"M / {m_r}")

    expr = gemm_mk.find(f"(_ + M / {m_r} * {m_r}) % {m_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, expr_to_string(expr.lhs().lhs()))

    if not isTransA:
        gemm_mk = apply(lift_scope)(gemm_mk, gemm_mk.find_loop("i1", many=True))
    gemm_mk, _ = extract_subproc(
        gemm_mk, gemm_mk.find_loop("i0o" if not isTransA else "i0").expand(0, 1 - int(isTransA)), gemm_mk.name() + "_A_pack"
    )
    isTransB = TransB == CblasTransValue
    packed_B_shape = ((1 - int(isTransB), ceil(max_N / n_r)), (int(isTransB), max_K), (1 - int(isTransB), n_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "BT" if isTransB else "B", packed_B_shape, "packed_B", rc=1)

    expr = gemm_mk.find(f"(_ + N / {n_r} * {n_r}) / {n_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"N / {n_r}")

    expr = gemm_mk.find(f"(_ + N / {n_r} * {n_r}) % {n_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, expr_to_string(expr.lhs().lhs()))

    if not isTransB:
        gemm_mk = optimize_level_1(
            gemm_mk, gemm_mk.find_loop("i1i"), precision, machine, n_r_fac, vec_tail="perfect", inter_tail="cut"
        )
        gemm_mk = optimize_level_1(gemm_mk, gemm_mk.find_loop("i1i"), precision, machine, 1)
        gemm_mk = unroll_and_jam(gemm_mk, gemm_mk.find_loop("i0"), 4, unroll=(0, 1, 0))
        gemm_mk = tile_loops_bottom_up(gemm_mk, gemm_mk.find_loop("i0o"), (17, 3))

    gemm_mk = set_memory(gemm_mk, cursors.alloc, ALIGNED_DRAM_STATIC)
    block = cursors.load.as_block()
    gemm_mk, _ = extract_subproc(gemm_mk, gemm_mk.forward(block).expand(0, 2 - 2 * int(isTransB)), gemm_mk.name() + "_B_pack")
    gemm_mk = extract_and_schedule(schedule_compute)(
        gemm_mk, i_loop, gemm_mk.name() + "_compute", precision, machine, m_r, n_r_fac
    )
    return gemm_mk_starter, simplify(gemm_mk)


def schedule_tiled(gemm, i_loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile, TransA, TransB):
    gemm_macro = schedule_macro(gemm, precision, machine, M_tile, N_tile, K_tile, m_r, n_r_fac, TransA, TransB)

    k_loop = get_inner_loop(gemm, get_inner_loop(gemm, i_loop))
    gemm_tiled = repeate_n(lift_scope)(gemm, k_loop, n=1)
    gemm_tiled = tile_loops_bottom_up(gemm_tiled, i_loop, [M_tile, K_tile, N_tile])
    gemm_tiled = apply(repeate_n(reorder_loops))(gemm_tiled, gemm_tiled.find_loop("ki", many=True), n=1)
    gemm_tiled = replace_all_stmts(gemm_tiled, [gemm_macro])

    gemm_tiled = inline_calls(gemm_tiled, subproc=gemm_macro[1])

    gemm_tiled = apply(hoist_from_loop)(gemm_tiled, gemm_tiled.find_loop("jo", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_A : _", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_B : _", many=True))
    return simplify(gemm_tiled)


def schedule_gemm(gemm, i_loop, precision, machine, TransA=None, TransB=None):
    # Correct parameters
    # PARAMS = {"avx2": (4, 3, 427, 17, 344), "avx512": (8, 3, 427, 17, 344), "neon": (1, 1, 1, 1, 1)}

    # For fast compilation
    PARAMS = {"avx2": (2, 2, 427, 17, 344), "avx512": (2, 2, 427, 17, 344), "neon": (1, 1, 1, 1, 1)}
    m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[machine.name]

    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    M_tile = M_tile_fac * m_r
    N_tile = N_tile_fac * n_r

    starter_gemm = gemm
    gemm = specialize(gemm, i_loop, f"N <= 100")
    if_c = gemm.forward(i_loop.as_block())[0]
    if TransB == CblasNoTransValue:
        gemm = extract_and_schedule(schedule_compute)(
            gemm, if_c.body()[0], gemm.name() + "_small", precision, machine, m_r, n_r_fac
        )
    tiled_gemm = schedule_tiled(
        starter_gemm, starter_gemm.body()[0], precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile, TransA, TransB
    )
    tiled_gemm = rename(tiled_gemm, tiled_gemm.name() + "_tiled")
    gemm = replace_all_stmts(gemm, (starter_gemm, tiled_gemm))
    return gemm


variants_generator(schedule_gemm, targets=("avx2", "avx512"))(gemm, "i", globals=globals())
