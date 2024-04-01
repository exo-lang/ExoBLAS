from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *


@proc
def trmv_rm_un(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        xCopy[n - i - 1] = 0.0
        for j in seq(0, i):
            xCopy[n - i - 1] += A[n - i - 1, n - j - 1] * x[n - j - 1]
        if Diag == 0:
            xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
        else:
            xCopy[n - i - 1] += x[n - i - 1]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_ln(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]

    for i in seq(0, n):
        xCopy[i] = 0.0
        for j in seq(0, i):
            xCopy[i] += A[i, j] * x[j]
        if Diag == 0:
            xCopy[i] += A[i, i] * x[i]
        else:
            xCopy[i] += x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_ut(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for l in seq(0, n):
        xCopy[l] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[n - j - 1] += A[n - i - 1, n - j - 1] * x[n - i - 1]
        if Diag == 0:
            xCopy[n - i - 1] += A[n - i - 1, n - i - 1] * x[n - i - 1]
        else:
            xCopy[n - i - 1] += x[n - i - 1]

    for l in seq(0, n):
        x[l] = xCopy[l]


@proc
def trmv_rm_lt(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    xCopy: R[n]
    for l in seq(0, n):
        xCopy[l] = 0.0

    for i in seq(0, n):
        for j in seq(0, i):
            xCopy[j] += A[i, j] * x[i]
        if Diag == 0:
            xCopy[i] += A[i, i] * x[i]
        else:
            xCopy[i] += x[i]

    for l in seq(0, n):
        x[l] = xCopy[l]


for proc in trmv_rm_un, trmv_rm_ln, trmv_rm_ut, trmv_rm_lt:
    variants_generator(optimize_level_2)(proc, "i", 4, 2, round_up=False, globals=globals())
