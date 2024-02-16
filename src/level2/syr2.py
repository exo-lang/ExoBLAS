from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *


### EXO_LOC ALGORITHM START ###
@proc
def syr2_rm_u(n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, i + 1):
            A[n - i - 1, n - j - 1] += (alpha * x[n - i - 1]) * y[n - j - 1] + (
                alpha * y[n - i - 1]
            ) * x[n - j - 1]


@proc
def syr2_rm_l(n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        for j in seq(0, i + 1):
            A[i, j] += (alpha * x[i]) * y[j] + (alpha * y[i]) * x[j]


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###

variants_generator(optimize_level_2)(
    syr2_rm_u, "i", 4, 1, round_up=False, globals=globals()
)
variants_generator(optimize_level_2)(
    syr2_rm_l, "i", 4, 1, round_up=False, globals=globals()
)

### EXO_LOC SCHEDULE END ###
