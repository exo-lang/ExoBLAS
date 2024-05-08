from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *


@proc
def syr2_rm(
    Uplo: size,
    n: size,
    alpha: R,
    x: [R][n],
    x_copy: [R][n],
    y: [R][n],
    y_copy: [R][n],
    A: [R][n, n],
):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, i + 1):
            if Uplo == CblasUpperValue:
                A[n - i - 1, n - j - 1] += (alpha * x[n - i - 1]) * y[n - j - 1] + (alpha * y_copy[n - i - 1]) * x_copy[n - j - 1]
            else:
                A[i, j] += (alpha * x[i]) * y[j] + (alpha * y_copy[i]) * x_copy[j]


variants_generator(optimize_level_2)(syr2_rm, "i", 4, 2, globals=globals())
