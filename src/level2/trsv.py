from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *
from composed_schedules import *


@proc
def trsv_rm_un(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
        pivot: R
        if Diag == 0:
            pivot = A[n - i - 1, n - i - 1]
        else:
            pivot = 1.0
        x[n - i - 1] = (x[n - i - 1] - dot) / pivot


@proc
def trsv_rm_ln(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    for i in seq(0, n):
        dot: R
        dot = 0.0
        for j in seq(0, i):
            dot += A[i, j] * x[j]

        pivot: R
        if Diag == 0:
            pivot = A[i, i]
        else:
            pivot = 1.0

        x[i] = (x[i] - dot) / pivot


@proc
def trsv_rm_ut(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1
    dot: R[n]
    for i in seq(0, n):
        dot[i] = 0.0
    for i in seq(0, n):
        pivot: R
        if Diag == 0:
            pivot = A[i, i]
        else:
            pivot = 1.0
        x[i] = (x[i] - dot[i]) / pivot
        for j in seq(0, n - i - 1):
            dot[i + j + 1] += A[i, i + j + 1] * x[i]


@proc
def trsv_rm_lt(Diag: index, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1
    dot: R[n]
    for i in seq(0, n):
        dot[i] = 0.0
    for i in seq(0, n):
        pivot: R
        if Diag == 0:
            pivot = A[n - i - 1, n - i - 1]
        else:
            pivot = 1.0
        x[n - i - 1] = (x[n - i - 1] - dot[n - i - 1]) / pivot
        for j in seq(0, n - i - 1):
            dot[j] += A[n - i - 1, j] * x[n - i - 1]


variants_generator(optimize_level_2)(
    trsv_rm_un, "i", 4, 2, round_up=False, globals=globals()
)
variants_generator(optimize_level_2)(
    trsv_rm_ln, "i", 4, 2, round_up=False, globals=globals()
)
variants_generator(optimize_level_1)(
    trsv_rm_ut, "j", 4, vec_tail="cut", globals=globals()
)
variants_generator(optimize_level_1)(
    trsv_rm_lt, "j", 4, vec_tail="cut", globals=globals()
)
