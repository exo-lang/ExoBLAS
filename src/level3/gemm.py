from __future__ import annotations
import math

from exo import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *
from exo.libs.memories import DRAM_STATIC

import exo_blas_config as C
from stdlib import *
from codegen_helpers import *
from blaslib import *
from cblas_enums import *


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


def schedule_compute(gemm_compute, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac
    i_loop = gemm_compute.body()[0]
    j_loop = get_inner_loop(gemm_compute, i_loop)
    k_loop = get_inner_loop(gemm_compute, j_loop)
    gemm_compute = auto_stage_mem(gemm_compute, gemm_compute.body(), "alpha", "alpha_")
    gemm_compute, cs = auto_stage_mem(gemm_compute, k_loop, "C", "C_tile", accum=True, rc=1)
    gemm_compute = lift_reduce_constant(gemm_compute, cs.load.expand(0, 1))
    assign = gemm_compute.forward(cs.store).prev()
    gemm_compute = inline_assign(gemm_compute, assign)
    gemm_compute = set_memory(gemm_compute, cs.alloc, machine.mem_type)

    gemm_compute = tile_loops_bottom_up(gemm_compute, i_loop, (m_r, n_r, None), tail="guard")
    gemm_compute = repeate_n(parallelize_and_lift_alloc)(gemm_compute, cs.alloc, n=4)
    gemm_compute = fission(gemm_compute, gemm_compute.forward(cs.load).after(), n_lifts=4)
    gemm_compute = fission(gemm_compute, gemm_compute.forward(cs.store).before(), n_lifts=4)
    gemm_compute = repeate_n(lift_scope)(gemm_compute, k_loop, n=4)
    gemm_compute = divide_dim(gemm_compute, cs.alloc, 1, vw)
    init_i, cmp_i, axpy_i = gemm_compute.find_loop("ii", many=True)
    init_j, cmp_j, axpy_j = gemm_compute.find_loop("ji", many=True)
    gemm_compute = vectorize(gemm_compute, init_j, vw, precision, machine.mem_type, tail="perfect")
    gemm_compute = unroll_loop(gemm_compute, init_j)
    gemm_compute, (o_cmp_j, i_cmp_j, _) = divide_loop_(gemm_compute, cmp_j, vw, tail="perfect", rc=True)
    gemm_compute = simplify(gemm_compute)
    gemm_compute, cursors = auto_stage_mem(gemm_compute, cmp_i, "packed_B", rc=True)
    gemm_compute = set_memory(gemm_compute, cursors.alloc, machine.mem_type)
    gemm_compute = simplify(gemm_compute)
    gemm_compute = divide_dim(gemm_compute, cursors.alloc, 0, vw)
    gemm_compute = vectorize(gemm_compute, cursors.load, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    gemm_compute = unroll_loop(gemm_compute, cursors.load)
    gemm_compute = lift_scope(gemm_compute, i_cmp_j)
    i_cmp_j = gemm_compute.forward(i_cmp_j)
    gemm_compute = auto_stage_mem(gemm_compute, i_cmp_j.body(), "packed_A", "A_reg")
    gemm_compute = unroll_loop(gemm_compute, o_cmp_j)
    gemm_compute = vectorize(gemm_compute, i_cmp_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")

    gemm_compute = unroll_loop(gemm_compute, i_cmp_j)
    gemm_compute, alpah_cursors = auto_stage_mem(gemm_compute, axpy_j.body(), "alpha_", rc=True)
    gemm_compute = vectorize(gemm_compute, axpy_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    gemm_compute = simplify(interleave_loop(gemm_compute, axpy_j, n_r_fac))
    gemm_compute = unroll_loop(gemm_compute, axpy_j)
    gemm_compute = simplify(gemm_compute)

    gemm_compute = specialize(gemm_compute, gemm_compute.body(), f"M % {m_r} == 0 and N % {n_r} == 0")

    def cut(proc, loop, cond, rng):
        loop = proc.forward(loop)
        cut_val = FormattedExprStr(f"_ - 1", loop.hi())
        proc, (loop1, loop2) = cut_loop_(proc, loop, cut_val, rc=True)
        proc = specialize(proc, loop2.body(), [f"{cond(loop2, i)} == {i}" for i in rng])
        return proc

    right_cond = lambda l, i: f"(N - {l.name()} * {n_r} + {vw - 1}) / {vw}"
    gemm_compute = cut(gemm_compute, gemm_compute.find_loop("jo #1"), right_cond, range(1, n_r_fac))

    gemm_compute = dce(gemm_compute)
    gemm_compute = replace_all_stmts(gemm_compute, machine.get_instructions(precision))

    gemm_compute = simplify(unroll_loops(gemm_compute))
    bottom_cond = lambda l, i: f"M - {l.name()} * {m_r}"
    gemm_compute = cut(gemm_compute, gemm_compute.find_loop("io #1"), bottom_cond, range(m_r, 1, -1))

    def rewrite(p):
        try:
            p = delete_pass(p)
        except:
            pass
        p = dce(p)
        p = divide_loop_(p, p.find_loop("k"), 4, tail="cut")
        p = unroll_loop(p, p.find_loop("ki"))
        return simplify(p)

    blocks = gemm_compute.find_loop("C_tile:_", many=True)
    for i, tile in enumerate(blocks):
        name = gemm_compute.name() + str(i)
        gemm_compute = extract_and_schedule(rewrite)(gemm_compute, tile.expand(), name)
    return simplify(gemm_compute)


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
    gemm_mk = set_memory(gemm_mk, cursors.alloc, DRAM_STATIC)

    expr = gemm_mk.find(f"(i0i + M / {m_r} * {m_r}) / {m_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"M / {m_r}")

    expr = gemm_mk.find(f"(i0i + M / {m_r} * {m_r}) % {m_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"i0i")
    gemm_mk = apply(lift_scope)(gemm_mk, gemm_mk.find_loop("i1", many=True))
    gemm_mk, _ = extract_subproc(gemm_mk, gemm_mk.find_loop("i0o").expand(0, 1), gemm_mk.name() + "_A_pack")
    packed_B_shape = ((1, max_N // n_r), (0, max_K), (1, n_r))
    gemm_mk, cursors = pack_mem(gemm_mk, i_loop, "B", packed_B_shape, "packed_B", rc=1)

    expr = gemm_mk.find(f"(i1i + N / {n_r} * {n_r}) / {n_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"N / {n_r}")

    expr = gemm_mk.find(f"(i1i + N / {n_r} * {n_r}) % {n_r}")
    gemm_mk = rewrite_expr(gemm_mk, expr, f"i1i")
    gemm_mk = optimize_level_1(
        gemm_mk, gemm_mk.find_loop("i1i"), precision, machine, n_r_fac, vec_tail="perfect", inter_tail="cut"
    )
    gemm_mk = optimize_level_1(gemm_mk, gemm_mk.find_loop("i1i"), precision, machine, 1)

    gemm_mk = set_memory(gemm_mk, cursors.alloc, DRAM_STATIC)
    gemm_mk, _ = extract_subproc(gemm_mk, cursors.load, gemm_mk.name() + "_B_pack")

    gemm_mk = extract_and_schedule(schedule_compute)(
        gemm_mk, i_loop, gemm_mk.name() + "_compute", precision, machine, m_r, n_r_fac
    )
    return gemm_mk_starter, simplify(gemm_mk)


def schedule(main_gemm, i_loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile, TransA=None, TransB=None):
    if TransA != CblasNoTransValue or TransB != CblasNoTransValue:
        return main_gemm

    gemm_macro = schedule_macro(main_gemm, precision, machine, M_tile, N_tile, K_tile, m_r, n_r_fac)

    gemm_tiled = main_gemm
    k_loop = get_inner_loop(gemm_tiled, get_inner_loop(gemm_tiled, i_loop))
    gemm_tiled = repeate_n(lift_scope)(gemm_tiled, k_loop, n=2)
    gemm_tiled = tile_loops_bottom_up(gemm_tiled, k_loop, [K_tile, M_tile, N_tile])
    gemm_tiled = apply(repeate_n(reorder_loops))(gemm_tiled, gemm_tiled.find_loop("ki", many=True), n=2)
    gemm_tiled = replace_all_stmts(gemm_tiled, [gemm_macro])

    gemm_tiled = inline_calls(gemm_tiled, subproc=gemm_macro[1])

    gemm_tiled = apply(hoist_from_loop)(gemm_tiled, gemm_tiled.find_loop("jo", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_A : _", many=True))
    gemm_tiled = squash_buffers(gemm_tiled, gemm_tiled.find("packed_B : _", many=True))
    return simplify(gemm_tiled)


PARAMS = {AVX2: (4, 3, 66, 3, 512), AVX512: (5, 5, 33, 2, 512), Neon: (1, 1, 1, 1, 1)}

m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[C.Machine.mem_type]
n_r = n_r_fac * C.Machine.vec_width("f32")


W_L1 = 12
S_L1 = 48 * 1024
C_L1 = 64
N_L1 = S_L1 // (C_L1 * W_L1)
C_Br = math.floor((W_L1 - 1) / (1 + m_r / n_r))
S_data = 4
K_tile = (C_Br * N_L1 * C_L1) // (n_r * S_data)

W_L2 = 20
S_L2 = 1280 * 1024
C_L2 = 64
N_L2 = S_L2 // (C_L2 * W_L2)
C_BT = math.floor((W_L2 - 1) / 2)
N_tile = (C_BT * N_L2 * C_L2) // (K_tile * S_data)

N_tile = (N_tile // n_r) * n_r

W_L3 = 12
S_L3 = 3 * 1024 * 1024
C_L3 = 64
N_L3 = S_L3 // (C_L3 * W_L3)
C_AT = math.floor((W_L3 - 1) / 2)
M_tile = (C_AT * N_L3 * C_L3) // (K_tile * S_data)

M_tile = (M_tile // m_r) * m_r

variants_generator(schedule, ("f32",), (AVX2, AVX512))(gemm, "i", m_r, n_r_fac, M_tile, N_tile, K_tile, globals=globals())
