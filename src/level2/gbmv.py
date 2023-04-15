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
def gbmv_row_major_template(
    m: size,
    n: size,
    kl: size,
    ku: size,
    alpha: f32,
    beta: f32,
    a: [f32][m, ku+kl+1],
    x: [f32][n],
    y: [f32][m],
):
    assert kl + 1 <= m
    assert ku + 1 <= n
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


def specialize_gbmv(precision):
    prefix = "s" if precision == "f32" else "d"
    name = gbmv_row_major_template.name().replace("_template", "")
    specialized = rename(gbmv_row_major_template, "exo_" + prefix + name)

    args = ["alpha", "beta", "x", "y", "a"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


exo_sgbmv_row_major = specialize_gbmv("f32")

if __name__ == "__main__":
    print(exo_sgbmv_row_major)


__all__ = ["exo_sgbmv_row_major"]
