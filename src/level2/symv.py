from __future__ import annotations

from exo import *

from codegen_helpers import *
from blaslib import *
from cblas_enums import *


@proc
def symv_rm(Uplo: size, n: size, alpha: R, A: [R][n, n], x: [R][n], beta: R, y: [R][n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        y[i] = beta * y[i]

    for i in seq(0, n):
        if Uplo == CblasUpperValue:
            temp: R
            temp = alpha * x[n - i - 1]
            dot: R
            dot = 0.0
            for j in seq(0, i + 1):
                y[n - j - 1] += temp * A[n - i - 1, n - j - 1]
                dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
            y[n - i - 1] += alpha * (dot - A[n - i - 1, n - i - 1] * x[n - i - 1])
        else:
            temp: R
            temp = alpha * x[i]
            dot: R
            dot = 0.0
            for j in seq(0, i + 1):
                y[j] += temp * A[i, j]
                dot += A[i, j] * x[j]
            y[i] += alpha * (dot - A[i, i] * x[i])


variants_generator(optimize_level_2)(symv_rm, "i #1", 4, 1, globals=globals())
