from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def gbmv(n : size, m : size, A : R[m, n], x : R[n], y : R[m], alpha : R, beta : R):
    for j in seq(0, m):
        for i in seq(0, n):
            y[j] += alpha * A[j, i] * x[i] + beta * y[j]

if __name__ == "__main__":
    print(gbmv)

__all__ = ["gbmv"]
