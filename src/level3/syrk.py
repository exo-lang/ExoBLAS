from __future__ import annotations

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

PARAMS = {AVX2: (4, 3, 66, 3, 512), AVX512: (6, 4, 44, 1, 512), Neon: (1, 1, 1, 1, 1)}

m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[C.Machine.mem_type]
n_r = n_r_fac * C.Machine.vec_width("f32")
M_tile = M_tile_fac * m_r
N_tile = N_tile_fac * n_r

variants_generator(identity_schedule, ("f32",), (AVX2, AVX512))(
    syrk_rm_l, "i", m_r, n_r_fac, M_tile, N_tile, K_tile, globals=globals()
)
variants_generator(identity_schedule, ("f32",), (AVX2, AVX512))(
    syrk_rm_u, "i", m_r, n_r_fac, M_tile, N_tile, K_tile, globals=globals()
)
