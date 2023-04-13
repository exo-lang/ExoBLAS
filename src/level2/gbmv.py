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
    a: f32[m, n],
    x: f32[n],
    y: f32[m],
    kl: size,
    ku: size,
):
    for i in seq(0, m):
        y[i] = beta * y[i]
        for j in seq(0, n):
            if i - kl <= j and j <= i + ku:
                if 0 <= j + kl - i and j + kl - i < n:
                    y[i] += alpha * a[i, j + kl - i] * x[j]


@proc
def gbmv_no_trans_v2(
    alpha: f32,
    beta: f32,
    n: size,
    m: size,
    lda: size,
    kl: size,
    ku: size,
    a: f32[m, lda],
    x: f32[n],
    y: f32[m],
):
    # GBMV written with only one if check.
    # TODO: add transpose support
    assert ku + kl + 1 <= lda
    for i in seq(0, m):
        y[i] = beta * y[i]
        for j in seq(0, ku + kl + 1):
            if i + (j - kl) >= 0 and i + (j - kl) < n:
                y[i] += alpha * a[i, j] * x[i + (j - kl)]


@proc
def gbmv_trans(
    alpha: f32,
    beta: f32,
    n: size,
    m: size,
    a: f32[n, m],
    x: f32[n],
    y: f32[m],
    kl: size,
    ku: size,
):
    for i in seq(0, m):
        y[i] = beta * y[i]
        for j in seq(0, n):
            if i - ku <= j and j <= i + kl:
                if 0 <= j + kl - i and j + kl - i < n:
                    y[i] += alpha * a[j + kl - i, i] * x[j]


if __name__ == "__main__":
    print(gbmv_trans)
    print(gbmv_no_trans)
    print(gbmv_no_trans_v2)

__all__ = ["gbmv_trans", "gbmv_no_trans"]
