from __future__ import annotations

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
def symm_rm(Side: size, Uplo: size, M: size, N: size, alpha: R, Aleft: [R][M, M], Aright: [R][N, N], B: [R][M, N], C: [R][M, N]):
    for i in seq(0, M):
        for j in seq(0, N):
            if Side == CblasLeftValue:
                for k in seq(0, M):
                    a_val: R
                    if Uplo == CblasUpperValue:
                        if k < i + 1:
                            a_val = Aleft[k, i]
                        else:
                            a_val = Aleft[i, k]
                    else:
                        if k < i + 1:
                            a_val = Aleft[i, k]
                        else:
                            a_val = Aleft[k, i]
                    C[i, j] += alpha * (a_val * B[k, j])
            else:
                for k in seq(0, N):
                    a_val: R
                    if Uplo == CblasUpperValue:
                        if j < k + 1:
                            a_val = Aright[j, k]
                        else:
                            a_val = Aright[k, j]
                    else:
                        if j < k + 1:
                            a_val = Aright[k, j]
                        else:
                            a_val = Aright[j, k]
                    C[i, j] += alpha * (B[i, k] * a_val)


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
    gemm_compute, cursors = auto_stage_mem(gemm_compute, cmp_i, "B", rc=True)
    gemm_compute = set_memory(gemm_compute, cursors.alloc, machine.mem_type)
    gemm_compute = simplify(gemm_compute)
    gemm_compute = divide_dim(gemm_compute, cursors.alloc, 0, vw)
    gemm_compute = vectorize(gemm_compute, cursors.load, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    gemm_compute = unroll_loop(gemm_compute, cursors.load)
    gemm_compute = vectorize(gemm_compute, i_cmp_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
    gemm_compute = unroll_loop(gemm_compute, i_cmp_j)
    gemm_compute = unroll_loop(gemm_compute, o_cmp_j)
    gemm_compute, alpah_cursors = auto_stage_mem(gemm_compute, axpy_j.body(), "alpha", rc=True)
    gemm_compute = vectorize(gemm_compute, axpy_j, vw, precision, machine.mem_type, rules=[fma_rule], tail="perfect")
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
        return simplify(p)

    blocks = gemm_compute.find_loop("C_tile:_", many=True)
    for i, tile in enumerate(blocks):
        name = gemm_compute.name() + str(i)
        gemm_compute = extract_and_schedule(rewrite)(gemm_compute, tile.expand(), name)
    return simplify(gemm_compute)


def schedule(main_symm, i_loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile):
    gemm_macro_spec = specialize_precision(gemm, precision)
    gemm_macro = schedule_compute(gemm_macro_spec, precision, machine, m_r, n_r_fac)

    symm_tiled = main_symm
    k_loop = get_inner_loop(symm_tiled, get_inner_loop(symm_tiled, i_loop))
    symm_tiled = repeate_n(lift_scope)(symm_tiled, k_loop, n=2)
    symm_tiled = tile_loops_bottom_up(symm_tiled, k_loop, [K_tile, M_tile, N_tile])
    symm_tiled = apply(repeate_n(reorder_loops))(symm_tiled, symm_tiled.find_loop("ki", many=True), n=2)

    def pack_A(proc, block):
        block = proc.forward(block)
        alloc = block[0]
        loads = block[1]
        proc = parallelize_and_lift_alloc(proc, alloc)
        proc = lift_alloc(proc, alloc)
        proc = parallelize_and_lift_alloc(proc, alloc)
        proc = fission(proc, loads.after(), n_lifts=3)
        loads = proc.forward(loads)
        proc = remove_loop(proc, loads.parent().parent())
        proc = bound_alloc(proc, alloc, (M_tile, K_tile))
        proc = set_memory(proc, alloc, DRAM_STATIC)
        loads = proc.forward(loads)
        cond = loads.cond()
        return proc

    loops = symm_tiled.find_loop("ii", many=True)
    symm_tiled = apply(pack_mem)(symm_tiled, loops, "B", ((0, K_tile), (1, N_tile)), "packed_B")

    blocks = symm_tiled.find("a_val:_", many=True)
    blocks = [b.expand(0, 1) for b in blocks]
    symm_tiled = apply(pack_A)(symm_tiled, blocks)
    symm_tiled = replace_all_stmts(symm_tiled, [(gemm_macro_spec, gemm_macro)])

    # TODO: Look into why analysis is unsatisfiable here
    symm_tiled = apply(hoist_from_loop)(symm_tiled, symm_tiled.find_loop("jo", many=True))
    symm_tiled = squash_buffers(symm_tiled, symm_tiled.find("a_val : _", many=True))
    buffers = symm_tiled.find("packed_B : _", many=True)
    symm_tiled = apply(set_memory)(symm_tiled, buffers, DRAM_STATIC)
    symm_tiled = squash_buffers(symm_tiled, buffers)
    return simplify(symm_tiled)


def schedule_symm(symm, loop, precision, machine, Side=None, Uplo=None):
    PARAMS = {"avx2": (2, 2, 66, 3, 512), "avx512": (6, 4, 44, 1, 512), "neon": (1, 1, 1, 1, 1)}

    m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[machine.name]
    n_r = n_r_fac * machine.vec_width("f32")

    M_tile = M_tile_fac * m_r
    N_tile = N_tile_fac * n_r

    if Side == CblasRightValue or Uplo == CblasUpperValue:
        return symm

    symm = schedule(symm, loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile)
    return symm


variants_generator(schedule_symm, ("f32",), ("avx2", "avx512"))(symm_rm, "i", globals=globals())
