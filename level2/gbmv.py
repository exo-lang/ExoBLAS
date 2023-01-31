from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

"""
GEMV with a general band matrix
"""

@proc
def gbmv_no_trans(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  A: f32[n, m],
  x: f32[m],
  y: f32[n],
  kl : size,
  ku : size
):
  for i in seq(0, n):
      y[i] = beta * y[i]
      for j in seq(0, m):
          if i-kl <= j and j <= i+ku:
              if 0 <= j + kl - i and j + kl - i < m:
                  y[i] += alpha * A[i, j + kl - i] * x[j]

@proc
def gbmv_trans(
  alpha: f32,
  beta: f32,
  n: size,
  m: size,
  A: f32[n, m],
  x: f32[n],
  y: f32[m],
  kl : size,
  ku : size
):
  for i in seq(0, m):
      y[i] = beta * y[i]
      for j in seq(0, n):
          if i-ku <= j and j <= i+kl:
              if 0 <= j + ku - i and j + ku - i < n:
                  y[i] += alpha * A[j + ku - i, i] * x[j]


if __name__ == "__main__":
    print(gbmv_trans)
    print(gbmv_no_trans)

__all__ = ["gbmv_trans", "gbmv_no_trans"]
