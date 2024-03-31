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
    Uplo: size, Trans: size, N: size, K: size, alpha: R, A: [R][N, K], B: [R][N, K], AT: [R][K, N], BT: [R][K, N], C: [R][N, N]
):

    for i in seq(0, N):
        for j in seq(0, N):
            if (Uplo == CblasUpperValue and j >= i) or (Uplo == CblasLowerValue and j < i + 1):
                for k in seq(0, K):
                    if Trans == CblasNoTransValue:
                        C[i, j] += alpha * (A[i, k] * B[j, k] + B[i, k] * A[j, k])
                    else:
                        C[i, j] += alpha * (AT[k, i] * BT[k, j] + BT[k, i] * AT[k, j])


def schedule(syr2k, loop, precision, machine, Uplo=None, Trans=None):
    return syr2k


variants_generator(identity_schedule)(syr2k_rm, "i", globals=globals())
