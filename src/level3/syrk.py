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


@proc
def syrk_rm_l(N: size, K: size, alpha: R, A: [R][N, K], A_alias: [R][N, K], C: [R][N, N]):
    assert stride(A, 1) == 1
    assert stride(A_alias, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, N):
        for j in seq(0, i + 1):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * A_alias[j, k])


@proc
def syrk_rm_u(N: size, K: size, alpha: R, A: [R][N, K], A_alias: [R][N, K], C: [R][N, N]):
    assert stride(A, 1) == 1
    assert stride(A_alias, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, N):
        for j in seq(i, N):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * A_alias[j, k])


syrk_rm_u = shift_loop(syrk_rm_u, "j", 0)


@proc
def syrk_gemm(M: size, N: size, K: size, alpha: R, A: [R][N, K], A_alias: [R][N, K], C: [R][N, N]):
    assert stride(A, 1) == 1
    assert stride(A_alias, 1) == 1
    assert stride(C, 1) == 1
    assert M <= N

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * A_alias[j, k])


def schedule_compute(compute, precision, machine, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac
    i_loop = compute.body()[0]
    j_loop = get_inner_loop(compute, i_loop)
    k_loop = get_inner_loop(compute, j_loop)
    compute, cs = auto_stage_mem(compute, k_loop, "C", "C_tile", accum=True, rc=1)
    compute = lift_reduce_constant(compute, cs.load.expand(0, 1))
    assign = compute.forward(cs.store).prev()
    compute = inline_assign(compute, assign)
    compute = set_memory(compute, cs.alloc, machine.mem_type)

    compute = tile_loops_bottom_up(compute, i_loop, (m_r, n_r, None), tail="guard")

    compute = repeate_n(lift_scope)(compute, k_loop, n=4)
    compute = divide_dim(compute, cs.alloc, 1, vw)
    init_i, cmp_i, axpy_i = compute.find_loop("ii", many=True)
    init_j, cmp_j, axpy_j = compute.find_loop("ji", many=True)
    compute = vectorize(compute, init_j, vw, precision, machine.mem_type, tail="perfect")
    compute = unroll_loop(compute, init_j)
    compute, (o_cmp_j, i_cmp_j, _) = divide_loop_(compute, cmp_j, vw, tail="perfect", rc=True)
    compute = simplify(compute)
    compute, cursors = auto_stage_mem(compute, cmp_i, "packed_A_alias", rc=True)
    compute = set_memory(compute, cursors.alloc, machine.mem_type)
    compute = simplify(compute)
    compute = divide_dim(compute, cursors.alloc, 0, vw)
    compute = vectorize(compute, cursors.load, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    compute = unroll_loop(compute, cursors.load)
    compute = vectorize(compute, i_cmp_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    compute = unroll_loop(compute, i_cmp_j)
    compute = unroll_loop(compute, o_cmp_j)
    compute, alpah_cursors = auto_stage_mem(compute, axpy_j.body(), "alpha", rc=True)
    compute = vectorize(compute, axpy_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    compute = unroll_loop(compute, axpy_j)
    compute = simplify(compute)

    def cut(proc, loop, cond, rng):
        loop = proc.forward(loop)
        cut_val = FormattedExprStr(f"_ - 1", loop.hi())
        proc, (loop1, loop2) = cut_loop_(proc, loop, cut_val, rc=True)
        proc = specialize(proc, loop2.body(), [f"{cond(loop2, i)} == {i}" for i in rng])
        return proc

    right_cond = lambda l, i: f"(N - {l.name()} * {n_r} + {vw - 1}) / {vw}"
    compute = cut(compute, j_loop, right_cond, range(1, n_r_fac))
    compute = dce(compute)
    compute = replace_all_stmts(compute, machine.get_instructions(precision))

    compute = simplify(unroll_loops(compute))
    bottom_cond = lambda l, i: f"N - {l.name()} * {m_r}"
    compute = cut(compute, i_loop, bottom_cond, range(m_r, 1, -1))

    def rewrite(p):
        try:
            p = delete_pass(p)
        except:
            pass
        p = dce(p)
        return simplify(p)

    blocks = compute.find_loop("C_tile:_", many=True)
    for i, tile in enumerate(blocks):
        name = compute.name() + str(i)
        compute = extract_and_schedule(rewrite)(compute, tile.expand(), name)
    return simplify(compute)


def schedule_macro(mk, precision, machine, max_N, max_K, m_r, n_r_fac):
    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    for var, max_var in zip(("N", "K"), (max_N, max_K)):
        mk = mk.add_assertion(f"{var} <= {max_var}")

    mk_starter = mk
    mk = rename(mk, mk.name() + "_mk")
    i_loop = mk.body()[0]
    packed_A_shape = ((0, max_N // m_r), (1, max_K), (0, m_r))
    mk, cursors = pack_mem(mk, i_loop, "A", packed_A_shape, "packed_A", rc=1)
    mk = set_memory(mk, cursors.alloc, DRAM_STATIC)
    mk, _ = extract_subproc(mk, cursors.load, mk.name() + "_A_pack")

    # TODO: This packing step is doing more work the necessary (packing the whole matrix, not jus triangle)
    packed_A_alias_shape = ((0, max_N // n_r), (1, max_K), (0, n_r))
    mk, cursors = pack_mem(mk, i_loop, "A_alias", packed_A_alias_shape, "packed_A_alias", rc=1)
    mk = set_memory(mk, cursors.alloc, DRAM_STATIC)
    mk, _ = extract_subproc(mk, cursors.load, mk.name() + "_A_alias_pack")
    mk = extract_and_schedule(schedule_compute)(mk, i_loop, mk.name() + "_compute", precision, machine, m_r, n_r_fac)
    return mk_starter, simplify(mk)


def schedule(proc, i_loop, precision, machine, m_r, n_r_fac, N_tile, K_tile):
    syrk_macro = schedule_macro(proc, precision, machine, N_tile, K_tile, m_r, n_r_fac)

    syrk_gemm_macro = specialize_precision(syrk_gemm, precision)
    syrk_gemm_macro = schedule_macro(syrk_gemm_macro, precision, machine, N_tile, K_tile, m_r, n_r_fac)

    tiled = proc
    j_loop = get_inner_loop(tiled, i_loop)
    k_loop = get_inner_loop(tiled, j_loop)
    tiled = repeate_n(lift_scope)(tiled, k_loop, n=2)
    tiled = tile_loops_bottom_up(tiled, k_loop, [K_tile, N_tile, N_tile], tail="guard")

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
    tiled = replace_all_stmts(tiled, [syrk_macro, syrk_gemm_macro])
    tiled = inline_calls(tiled, subproc=syrk_macro[1])
    tiled = inline_calls(tiled, subproc=syrk_gemm_macro[1])

    tiled = apply(hoist_from_loop)(tiled, tiled.find_loop("jo", many=True))
    tiled = squash_buffers(tiled, tiled.find("packed_A : _", many=True))
    tiled = squash_buffers(tiled, tiled.find("packed_A_alias : _", many=True))

    return simplify(tiled)


PARAMS = {AVX2: (2, 2, 32, 512), AVX512: (5, 5, 2, 512), Neon: (1, 1, 1, 1)}

m_r, n_r_fac, N_tile_fac, K_tile = PARAMS[C.Machine.mem_type]
n_r = n_r_fac * C.Machine.vec_width("f32")

lcm = (m_r * n_r) // math.gcd(m_r, n_r)
N_tile = lcm * N_tile_fac

variants_generator(schedule, ("f32",), (AVX2, AVX512))(syrk_rm_l, "i", m_r, n_r_fac, N_tile, K_tile, globals=globals())
variants_generator(identity_schedule, ("f32",), (AVX2, AVX512))(syrk_rm_u, "i", m_r, n_r_fac, N_tile, K_tile, globals=globals())
