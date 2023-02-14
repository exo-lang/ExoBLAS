from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

@proc
def ger(m: size, n: size, alpha: f32, x: [f32][m], incx: stride, y: [f32][n], incy: stride, A: [f32][m, n], lda: stride):
  assert stride(x, 0) == incx
  assert stride(y, 0) == incy
  assert stride(A, 0) == lda
  assert stride(A, 1) == 1
  
  for i in seq(0, m):
    for j in seq(0, n):
        A[i, j] += alpha * x[i] * y[j]

if __name__ == "__main__":
    print(ger)

__all__ = ["ger"]
