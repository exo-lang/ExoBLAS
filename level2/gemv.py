from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

"""
Possible cases:
  - transpose or not
  - column or row major storage
"""

@proc
def gemv(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  a: f32[m, n],
  x: f32[n],
  y: f32[m],
):
  for i in seq(0, m):
    y[i] = beta * y[i]
    for j in seq(0, n):
      y[i] += alpha * a[i, j] * x[j]


if __name__ == "__main__":
  print(gemv)

__all__ = ["gemv"]
