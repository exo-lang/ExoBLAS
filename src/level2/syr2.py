from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


# equivalent to lower triangular, column major
@proc
def syr2_upper(n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]):
  for i in seq(0, n):
    for j in seq(i, n):
        A[i, j] += alpha * x[i] * y[j] + alpha * y[i] * x[j]


# equivalent to upper triangular, row major
@proc
def syr2_lower(n: size, alpha: R, x: [R][n], y: [R][n], A: [R][n, n]):
  for i in seq(0, n):
    for j in seq(0, i+1):
        A[i, j] += alpha * x[i] * y[j] + alpha * y[i] * x[j]


def specialize_precision(precision):
    prefix = "s" if precision == "f32" else "d"
    specialized_copy = rename(syr2_upper, "exo_" + prefix + "syr2")
    for arg in ["alpha", "x", "y", "A"]:
        specialized_copy = set_precision(specialized_copy, arg, precision)
    return specialized_copy


ssyr2_upper = specialize_precision("f32")

if __name__ == "__main__":
    print(ssyr2_upper)

__all__ = ["ssyr2_upper"]
