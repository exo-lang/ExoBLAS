from __future__ import annotations

from exo import *

from codegen_helpers import *
from blaslib import *


@proc
def ger_rm(m: size, n: size, alpha: R, x: [R][m], y: [R][n], A: [R][m, n]):
    assert stride(A, 1) == 1

    for i in seq(0, m):
        for j in seq(0, n):
            A[i, j] += alpha * x[i] * y[j]


variants_generator(optimize_level_2)(ger_rm, "i", 4, 2, skinny_factor=(11, 4), globals=globals())
