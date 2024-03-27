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
def symm_rm_ll(M: size, N: size, alpha: R, A: [R][M, M], B: [R][M, N], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, M):
                a_val: R
                if k < i + 1:
                    a_val = A[i, k]
                else:
                    a_val = A[k, i]
                C[i, j] += alpha * (a_val * B[k, j])


@proc
def symm_rm_lu(M: size, N: size, alpha: R, A: [R][M, M], B: [R][M, N], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, M):
                a_val: R
                if k < i + 1:
                    a_val = A[k, i]
                else:
                    a_val = A[i, k]
                C[i, j] += alpha * (a_val * B[k, j])


@proc
def symm_rm_rl(M: size, N: size, alpha: R, A: [R][N, N], B: [R][M, N], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, N):
                a_val: R
                if j < k + 1:
                    a_val = A[k, j]
                else:
                    a_val = A[j, k]
                C[i, j] += alpha * (B[i, k] * a_val)


@proc
def symm_rm_ru(M: size, N: size, alpha: R, A: [R][N, N], B: [R][M, N], C: [R][M, N]):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, N):
                a_val: R
                if j < k + 1:
                    a_val = A[j, k]
                else:
                    a_val = A[k, j]
                C[i, j] += alpha * (B[i, k] * a_val)


variants_generator(identity_schedule)(symm_rm_ll, "i", globals=globals())
variants_generator(identity_schedule)(symm_rm_lu, "i", globals=globals())
variants_generator(identity_schedule)(symm_rm_rl, "i", globals=globals())
variants_generator(identity_schedule)(symm_rm_ru, "i", globals=globals())
