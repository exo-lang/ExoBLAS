from __future__ import annotations
import math
from math import ceil

from exo import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *
from exo.libs.memories import DRAM_STATIC

import exo_blas_config as C
from stdlib import *
from codegen_helpers import *
from blaslib import *
from memories import *


@proc
def syrk_rm(
    Uplo: size, Trans: size, N: size, K: size, alpha: R, A: [R][N, K], A_: [R][N, K], AT: [R][K, N], AT_: [R][K, N], C: [R][N, N]
):
    assert stride(A, 1) == 1
    assert stride(A_, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, N):
        for j in seq(0, N):
            if (Uplo == CblasUpperValue and j >= i) or (Uplo == CblasLowerValue and j < i + 1):
                for k in seq(0, K):
                    if Trans == CblasNoTransValue:
                        C[i, j] += alpha * (A[i, k] * A_[j, k])
                    else:
                        C[i, j] += alpha * (AT[k, i] * AT_[k, j])


@proc
def syrk_gemm(M: size, N: size, K: size, alpha: R, A: [R][N, K], A_: [R][N, K], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(A_, 1) == 1
    assert stride(C, 1) == 1
    assert M <= N

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * A_[j, k])


def schedule_macro(mk, precision, machine, max_N, max_K, m_r, n_r_fac, Trans):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    for var, max_var in zip(("N", "K"), (max_N, max_K)):
        mk = mk.add_assertion(f"{var} <= {max_var}")

    mk_starter = mk
    mk = rename(mk, mk.name() + "_mk")
    i_loop = mk.body()[0]

    isTrans = Trans == CblasTransValue
    packed_A_shape = ((isTrans, ceil(max_N / m_r)), (1 - isTrans, max_K), (isTrans, m_r))
    mk, cursors = pack_mem(mk, i_loop, mk.args()[3 + int("gemm" in mk.name())].name(), packed_A_shape, "packed_A", rc=1)
    mk = set_memory(mk, cursors.alloc, ALIGNED_DRAM_STATIC)
    mk, _ = extract_subproc(mk, cursors.load.as_block(), mk.name() + "_A_pack")

    # TODO: This packing step is doing more work the necessary (packing the whole matrix, not jus triangle)
    packed_A_alias_shape = ((isTrans, ceil(max_N / n_r)), (1 - isTrans, max_K), (isTrans, n_r))
    mk, cursors = pack_mem(mk, i_loop, mk.args()[4 + int("gemm" in mk.name())].name(), packed_A_alias_shape, "packed_A_", rc=1)
    mk = set_memory(mk, cursors.alloc, ALIGNED_DRAM_STATIC)
    mk, _ = extract_subproc(mk, cursors.load.as_block(), mk.name() + "_A_alias_pack")
    mk = extract_and_schedule(schedule_compute)(mk, i_loop, mk.name() + "_compute", precision, machine, m_r, n_r_fac)
    return mk_starter, simplify(mk)


def schedule(proc, i_loop, precision, machine, m_r, n_r_fac, N_tile, K_tile, Uplo=None, Trans=None):
    if Uplo != CblasLowerValue:
        return proc

    syrk_macro = schedule_macro(proc, precision, machine, N_tile, K_tile, m_r, n_r_fac, Trans)

    gemm = syrk_gemm
    if Trans == CblasTransValue:
        gemm = gemm.transpose(gemm.args()[4])
        gemm = gemm.transpose(gemm.args()[5])
        gemm = rename(gemm, gemm.name() + "_t")
    gemm = blas_specialize_precision(gemm, precision)
    gemm = schedule_macro(gemm, precision, machine, N_tile, K_tile, m_r, n_r_fac, Trans)

    tiled = proc
    j_loop = get_inner_loop(tiled, i_loop)
    k_loop = get_inner_loop(tiled, j_loop)
    tiled = repeate_n(lift_scope)(tiled, k_loop, n=2)
    tiled = tile_loops_bottom_up(tiled, k_loop, [K_tile, N_tile, N_tile], tail="guard1")

    # TODO: This code should be a part of tiling
    def rewrite(proc, loop):
        loop = proc.forward(loop)
        cut_point = FormattedExprStr("_ - 1", loop.hi())
        proc, (loop, loop2) = cut_loop_(proc, loop, cut_point, rc=True)
        proc = simplify(shift_loop(proc, loop2, 0))
        proc, success = attempt(unroll_loop)(proc, loop2, rs=True)
        if not success:
            proc = rewrite_expr(proc, loop2.hi(), "1")
            proc = unroll_loop(proc, loop2)
        return proc

    tiled = apply(rewrite)(tiled, (j_loop, i_loop, k_loop))
    tiled = simplify(dce(tiled))
    tiled = apply(attempt(bound_loop_by_if))(tiled, tiled.find_loop("ki", many=True))
    tiled = apply(attempt(bound_loop_by_if))(tiled, tiled.find_loop("ii", many=True))
    tiled = apply(attempt(bound_loop_by_if))(tiled, tiled.find_loop("ji", many=True))
    tiled = simplify(delete_pass(tiled))

    tiled = apply(repeate_n(reorder_loops))(tiled, tiled.find_loop("ki", many=True), n=2)
    tiled = replace_all_stmts(tiled, [syrk_macro, gemm])
    tiled = inline_calls(tiled, subproc=syrk_macro[1])
    tiled = inline_calls(tiled, subproc=gemm[1])

    tiled = apply(hoist_from_loop)(tiled, tiled.find_loop("jo", many=True))
    tiled = squash_buffers(tiled, tiled.find("packed_A : _", many=True))
    tiled = squash_buffers(tiled, tiled.find("packed_A_ : _", many=True))

    return simplify(tiled)


PARAMS = {"avx2": (4, 3, 32, 512), "avx512": (2, 2, 2, 512), "neon": (1, 1, 1, 1)}

m_r, n_r_fac, N_tile_fac, K_tile = PARAMS[C.Machine.name]
n_r = n_r_fac * C.Machine.vec_width("f32")
K_tile = 344
lcm = (m_r * n_r) // math.gcd(m_r, n_r)
N_tile = (408 // lcm) * lcm

variants_generator(schedule, targets=("avx2", "avx512"))(syrk_rm, "i", m_r, n_r_fac, N_tile, K_tile, globals=globals())
