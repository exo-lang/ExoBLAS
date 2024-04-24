from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *
from cblas_enums import *


@proc
def gemv_rm(TransA: size, m: size, n: size, alpha: R, beta: R, A: [R][m, n], AT: [R][n, m], x: [R][n], y: [R][m]):
    assert stride(A, 1) == 1
    for i in seq(0, m):
        y[i] = y[i] * beta
    if TransA == CblasNoTransValue:
        for i in seq(0, m):
            for j in seq(0, n):
                y[i] += alpha * (x[j] * A[i, j])
    else:
        for i in seq(0, n):
            for j in seq(0, m):
                y[j] += alpha * x[i] * AT[i, j]


variants_generator(optimize_level_2)(gemv_rm, "i #1", 4, 2, globals=globals())
