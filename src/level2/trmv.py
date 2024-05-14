from __future__ import annotations

from exo import *

from codegen_helpers import *
from blaslib import *


@proc
def trmv_rm(Uplo: size, TransA: size, Diag: size, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for l in seq(0, n):
        xCopy[l] = 0.0

    for i in seq(0, n):
        if TransA == CblasNoTransValue:
            if Uplo == CblasUpperValue:
                for j in seq(0, i):
                    xCopy[n - i - 1] += A[n - i - 1, n - j - 1] * x[n - j - 1]
                if Diag == CblasNonUnitValue:
                    xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
                else:
                    xCopy[n - i - 1] += x[n - i - 1]
            else:
                for j in seq(0, i):
                    xCopy[i] += A[i, j] * x[j]
                if Diag == CblasNonUnitValue:
                    xCopy[i] += A[i, i] * x[i]
                else:
                    xCopy[i] += x[i]
        else:
            if Uplo == CblasUpperValue:
                for j in seq(0, i):
                    xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]
                if Diag == CblasNonUnitValue:
                    xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
                else:
                    xCopy[n - i - 1] += x[n - i - 1]
            else:
                for j in seq(0, i):
                    xCopy[j] += A[i, j] * x[i]
                if Diag == CblasNonUnitValue:
                    xCopy[i] += A[i, i] * x[i]
                else:
                    xCopy[i] += x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


variants_generator(optimize_level_2)(trmv_rm, "i", 4, 2, globals=globals())
