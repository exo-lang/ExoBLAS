from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *
from cblas_enums import *


@proc
def gemv_rm(TransA: size, M: size, N: size, alpha: R, beta: R, A: [R][M, N], AT: [R][N, M], x: [R][N], y: [R][M]):
    assert stride(A, 1) == 1
    for i in seq(0, M):
        y[i] = y[i] * beta
    if TransA == CblasNoTransValue:
        for i in seq(0, M):
            for j in seq(0, N):
                y[i] += alpha * (x[j] * A[i, j])
    else:
        for i in seq(0, N):
            for j in seq(0, M):
                y[j] += alpha * x[i] * AT[i, j]


variants_generator(optimize_level_2)(gemv_rm, "i #1", 4, 2, skinny_factor=(11, 4), globals=globals())
