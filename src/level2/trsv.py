from __future__ import annotations

from exo import *

from blaslib import *
from codegen_helpers import *
from stdlib import *


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
            dot[n - j - 1] += A[i, n - j - 1] * x[i]


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


def schedule_t(proc, i_loop, precision, machine, rows_factor, cols_factor):
    i_loop = proc.forward(i_loop)
    vw = machine.vec_width(precision)
    proc, (tail_loop, i_loop) = cut_loop_(proc, i_loop, f"n % {rows_factor}", rc=True)
    proc = shift_loop(proc, i_loop, 0)
    j_loop = get_inner_loop(proc, i_loop)
    proc, (_, oi_loop, t_loop) = divide_loop_(proc, i_loop, rows_factor, tail="cut", rc=True)
    proc = round_loop(proc, j_loop, vw, up=False)
    proc = dce(simplify(proc))
    j_loop = proc.forward(j_loop)
    proc = rewrite_expr(proc, j_loop.hi().lhs(), f"(n - ({rows_factor} * io + n % {rows_factor}) - 1) / {vw}")
    j_loop_tail = j_loop.next()
    proc = reorder_stmt_forward(proc, j_loop)
    proc = optimize_level_1(proc, j_loop_tail, precision, machine, 1, vec_tail="predicate")
    proc = unroll_and_jam_parent(proc, j_loop, rows_factor)
    proc = optimize_level_1(proc, j_loop, precision, machine, cols_factor)
    tail_loop = get_inner_loop(proc, tail_loop)
    proc = optimize_level_1(proc, tail_loop, precision, machine, rows_factor * cols_factor)
    return proc


for trsv in trsv_rm_ut, trsv_rm_lt:
    variants_generator(schedule_t)(trsv, "i #1", 4, 2, globals=globals())

for trsv in trsv_rm_un, trsv_rm_ln:
    variants_generator(optimize_level_2)(trsv, "i", 4, 2, round_up=False, globals=globals())
