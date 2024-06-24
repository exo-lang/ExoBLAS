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
def syr2k_rm(
    Uplo: size,
    Trans: size,
    N: size,
    K: size,
    alpha: R,
    A: [R][N, K],
    A_: [R][N, K],
    B: [R][N, K],
    B_: [R][N, K],
    AT: [R][K, N],
    AT_: [R][K, N],
    BT: [R][K, N],
    BT_: [R][K, N],
    C: [R][N, N],
):

    for i in seq(0, N):
        for j in seq(0, N):
            if (Uplo == CblasUpperValue and j >= i) or (Uplo == CblasLowerValue and j < i + 1):
                for k in seq(0, K):
                    if Trans == CblasNoTransValue:
                        C[i, j] += alpha * (A[i, k] * B[j, k] + B_[i, k] * A_[j, k])
                    else:
                        C[i, j] += alpha * (AT[k, i] * BT[k, j] + BT_[k, i] * AT_[k, j])


def schedule(syr2k, loop, precision, machine, Uplo=None, Trans=None):
    return syr2k
    if Uplo != CblasLowerValue or Trans != CblasNoTransValue:
        return syr2k

    PARAMS = {"avx2": (2, 2, 427, 17, 344), "avx512": (8, 3, 33, 2, 512), "neon": (1, 1, 1, 1, 1)}
    m_r, n_r_fac, M_tile_fac, N_tile_fac, K_tile = PARAMS[machine.name]
    if precision == "f64":
        K_tile //= 2

    vw = machine.vec_width(precision)
    n_r = vw * n_r_fac

    M_tile = M_tile_fac * m_r
    N_tile = N_tile_fac * n_r

    return schedule_compute(syr2k, syr2k.body()[0], precision, machine, m_r, n_r_fac)


variants_generator(schedule, targets=("avx2", "avx512"))(syr2k_rm, "i", globals=globals())
