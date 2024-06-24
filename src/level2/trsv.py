from __future__ import annotations

from exo import *

from codegen_helpers import *
from blaslib import *
from stdlib import *


@proc
def trsv_rm(Uplo: size, TransA: size, Diag: size, n: size, x: [R][n], A: [R][n, n]):
    assert stride(A, 1) == 1

    Dot: R[n]
    if TransA == CblasNoTransValue:
        pass
    else:
        for l in seq(0, n):
            Dot[l] = 0.0

    for i in seq(0, n):
        if TransA == CblasNoTransValue:
            dot: R
            dot = 0.0
            if Uplo == CblasUpperValue:
                for j in seq(0, i):
                    dot += A[n - i - 1, n - j - 1] * x[n - j - 1]
                x[n - i - 1] = x[n - i - 1] - dot
                if Diag == CblasNonUnitValue:
                    x[n - i - 1] = x[n - i - 1] / A[n - i - 1, n - i - 1]
            else:
                for j in seq(0, i):
                    dot += A[i, j] * x[j]
                x[i] = x[i] - dot
                if Diag == CblasNonUnitValue:
                    x[i] = x[i] / A[i, i]

        else:
            if Uplo == CblasUpperValue:
                x[i] = x[i] - Dot[i]
                if Diag == CblasNonUnitValue:
                    x[i] = x[i] / A[i, i]

                for j in seq(0, n - i - 1):
                    Dot[n - j - 1] += A[i, n - j - 1] * x[i]
            else:
                x[n - i - 1] = x[n - i - 1] - Dot[n - i - 1]
                if Diag == CblasNonUnitValue:
                    x[n - i - 1] = x[n - i - 1] / A[n - i - 1, n - i - 1]

                for j in seq(0, n - i - 1):
                    Dot[j] += A[n - i - 1, j] * x[n - i - 1]


def schedule_t(proc, i_loop, precision, machine, rows_factor, cols_factor):
    i_loop = proc.forward(i_loop)
    vw = machine.vec_width(precision)
    rows_factor = min(rows_factor, vw)
    proc, (tail_loop, i_loop) = cut_loop_(proc, i_loop, f"n % {rows_factor}", rc=True)
    proc = shift_loop(proc, i_loop, 0)
    j_loop = get_inner_loop(proc, i_loop)
    proc = round_loop(proc, j_loop, vw, up=False)
    proc, (_, oi_loop, t_loop) = divide_loop_(proc, i_loop, rows_factor, tail="cut", rc=True)
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


def schedule(trsv, loop, precision, machine, Uplo=None, TransA=None):
    if TransA == CblasNoTransValue:
        return optimize_level_2(trsv, loop, precision, machine, 4, 2, round_up=False)
    else:
        if machine.name in ("avx2", "avx512"):
            trsv = schedule_t(trsv, loop, precision, machine, 4, 2)
        return trsv


variants_generator(schedule)(trsv_rm, "i", globals=globals())
