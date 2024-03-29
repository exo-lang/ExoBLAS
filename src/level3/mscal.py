from __future__ import annotations

from exo import *

import exo_blas_config as C
from stdlib import *
from codegen_helpers import *
from blaslib import *
from cblas_enums import *


@proc
def mscal_rm(M: size, N: size, alpha: R, A: [R][M, N]):
    assert stride(A, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            A[i, j] = A[i, j] * alpha


@proc
def trmscal_rm(Uplo: size, N: size, alpha: R, A: [R][N, N]):
    assert stride(A, 1) == 1

    for i in seq(0, N):
        for j in seq(0, N):
            if (Uplo == CblasUpperValue and j > i - 1) or ((Uplo > CblasUpperValue or Uplo < CblasUpperValue) and j < i + 1):
                A[i, j] = A[i, j] * alpha


def schedule_trmscal(trmscal, loop, precision, machine, Uplo=None):
    cut_point = "i" if Uplo == CblasUpperValue else "i + 1"
    trmscal, (loop1, loop2) = cut_loop_(trmscal, loop, cut_point, rc=True)
    trmscal = shift_loop(trmscal, loop2, 0)
    trmscal = simplify(delete_pass(dce(trmscal)))
    loop = loop2 if Uplo == CblasUpperValue else loop1
    trmscal = optimize_level_1(trmscal, loop, precision, machine, 4)
    return trmscal


variants_generator(schedule_trmscal)(trmscal_rm, "j", globals=globals())
variants_generator(optimize_level_1)(mscal_rm, "j", 4, globals=globals())
