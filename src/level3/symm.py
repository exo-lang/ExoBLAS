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
from memories import *


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
def gemm(M: size, N: size, K: size, alpha: R, A: [R][M, K], B: [R][N, K], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += alpha * (A[i, k] * B[j, k])


def schedule(main_symm, i_loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile, Side=None, Uplo=None):
    gemm_macro_spec = blas_specialize_precision(gemm, precision)
    sfx = ""
    sfx += "r" if Side == CblasRightValue else "l"
    sfx += "u" if Uplo == CblasUpperValue else "l"
    gemm_macro_spec = rename(gemm_macro_spec, gemm_macro_spec.name() + sfx)
    gemm_macro = schedule_compute(gemm_macro_spec, gemm_macro_spec.body()[0], precision, machine, m_r, n_r_fac)
    gemm_macro = gemm_macro_spec

    symm_tiled = main_symm
    k_loop = get_inner_loop(symm_tiled, get_inner_loop(symm_tiled, i_loop))
    symm_tiled = repeate_n(lift_scope)(symm_tiled, k_loop, n=1)
    symm_tiled = tile_loops_bottom_up(symm_tiled, i_loop, [M_tile, K_tile, N_tile])
    symm_tiled = apply(repeate_n(reorder_loops))(symm_tiled, symm_tiled.find_loop("ki", many=True), n=1)

    def pack_A(proc, block):
        block = proc.forward(block)
        alloc = block[0]
        loads = block[1]
        if Side == CblasLeftValue:
            proc = parallelize_and_lift_alloc(proc, alloc)
            proc = lift_alloc(proc, alloc)
            proc = parallelize_and_lift_alloc(proc, alloc)
        else:
            proc = parallelize_and_lift_alloc(proc, alloc)
            proc = parallelize_and_lift_alloc(proc, alloc)
            proc = lift_alloc(proc, alloc)
        proc = fission(proc, loads.after(), n_lifts=3)
        loads = proc.forward(loads)

        if Side == CblasLeftValue:
            proc = remove_loop(proc, loads.parent().parent())
            proc = bound_alloc(proc, alloc, (M_tile, K_tile))
        else:
            proc = remove_loop(proc, loads.parent().parent().parent())
            proc = bound_alloc(proc, alloc, (N_tile, K_tile))
        proc = set_memory(proc, alloc, ALIGNED_DRAM_STATIC)
        loads = proc.forward(loads)
        cond = loads.cond()
        return proc

    blocks = symm_tiled.find("a_val:_", many=True)
    blocks = [b.expand(0, 1) for b in blocks]
    symm_tiled = apply(pack_A)(symm_tiled, blocks)

    # TODO: Look into why analysis is unsatisfiable here
    symm_tiled = apply(hoist_from_loop)(symm_tiled, symm_tiled.find_loop("jo", many=True))
    symm_tiled = squash_buffers(symm_tiled, symm_tiled.find("a_val : _", many=True))

    loops = symm_tiled.find_loop("ii", many=True)[1::2]
    shape = ((1, N_tile), (0, K_tile)) if Side == CblasLeftValue else ((0, M_tile), (1, K_tile))
    symm_tiled = apply(pack_mem)(symm_tiled, loops, "B", shape, "packed_B")

    buffers = symm_tiled.find("packed_B : _", many=True)
    symm_tiled = apply(set_memory)(symm_tiled, buffers, ALIGNED_DRAM_STATIC)
    symm_tiled = squash_buffers(symm_tiled, buffers)
    symm_tiled = replace_all_stmts(symm_tiled, [(gemm_macro_spec, gemm_macro)])
    return simplify(symm_tiled)


def schedule_symm(symm, loop, precision, machine, Side=None, Uplo=None):
    # Correct parameters
    # PARAMS = {"avx2": (4, 3, 427, 17, 344), "avx512": (8, 3, 427, 17 // 2, 344), "neon": (1, 1, 1, 1, 1)}

    # For fast compilation
    PARAMS = {"avx2": (2, 2, 427, 17, 344), "avx512": (2, 2, 427, 17 // 2, 344), "neon": (1, 1, 1, 1, 1)}
    m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[machine.name]
    n_r = n_r_fac * machine.vec_width("f32")

    M_tile = M_tile_fac * m_r
    N_tile = N_tile_fac * n_r

    if Side == CblasRightValue:
        return symm

    symm = schedule(symm, loop, precision, machine, m_r, n_r_fac, M_tile, N_tile, K_tile, Side, Uplo)
    return symm


variants_generator(schedule_symm, targets=("avx2", "avx512"))(symm_rm, "i", globals=globals())
