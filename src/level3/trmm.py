from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

import exo_blas_config as C


@proc
def trmm_row_major_Left_Upper_NonTrans_template(
    m: size, n: size, alpha: R, A: [R][m, m], B: [R][m, n], Diag: size
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1

    for j in seq(0, n):
        for i in seq(0, m):
            if Diag == 0:
                B[i, j] = A[i, i] * B[i, j]
            for k in seq(0, m - i - 1):
                B[i, j] += A[i, i + k + 1] * B[i + k + 1, j]
            B[i, j] = alpha * B[i, j]


@proc
def trmm_row_major_Left_Lower_NonTrans_template(
    m: size, n: size, alpha: R, A: [R][m, m], B: [R][m, n], Diag: size
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1

    for j in seq(0, n):
        for i in seq(0, m):
            orgBij: R
            orgBij = B[m - i - 1, j]

            B[m - i - 1, j] = 0.0

            for k in seq(0, m - i - 1):
                B[m - i - 1, j] += A[m - i - 1, k] * B[k, j]

            if Diag == 0:
                B[m - i - 1, j] += A[m - i - 1, m - i - 1] * orgBij
            else:
                B[m - i - 1, j] += orgBij

            B[m - i - 1, j] = alpha * B[m - i - 1, j]


@proc
def trmm_row_major_Left_Upper_Trans_template(
    m: size, n: size, alpha: R, A: [R][m, m], B: [R][m, n], Diag: size
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1

    for j in seq(0, n):
        for i in seq(0, m):
            orgBij: R
            orgBij = B[m - i - 1, j]

            B[m - i - 1, j] = 0.0

            for k in seq(0, m - i - 1):
                B[m - i - 1, j] += A[k, m - i - 1] * B[k, j]

            if Diag == 0:
                B[m - i - 1, j] += A[m - i - 1, m - i - 1] * orgBij
            else:
                B[m - i - 1, j] += orgBij

            B[m - i - 1, j] = alpha * B[m - i - 1, j]


@proc
def trmm_row_major_Left_Lower_Trans_template(
    m: size, n: size, alpha: R, A: [R][m, m], B: [R][m, n], Diag: size
):
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1

    for j in seq(0, n):
        for i in seq(0, m):
            if Diag == 0:
                B[i, j] = A[i, i] * B[i, j]
            for k in seq(0, m - i - 1):
                B[i, j] += A[i + k + 1, i] * B[i + k + 1, j]
            B[i, j] = alpha * B[i, j]


def specialize_trmm(trmm, precision):
    prefix = "s" if precision == "f32" else "d"
    name = trmm.name()
    name = name.replace("_template", "")
    specialized = rename(trmm, "exo_" + prefix + name)

    args = ["alpha", "A", "B"]

    for arg in args:
        specialized = set_precision(specialized, arg, precision)

    return specialized


#################################################
# Kernel Parameters
#################################################

#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_strmm_row_major_Left_Upper_NonTrans = specialize_trmm(
    trmm_row_major_Left_Upper_NonTrans_template, "f32"
)
exo_strmm_row_major_Left_Upper_NonTrans = rename(
    exo_strmm_row_major_Left_Upper_NonTrans,
    exo_strmm_row_major_Left_Upper_NonTrans.name() + "",
)
exo_strmm_row_major_Left_Lower_NonTrans = specialize_trmm(
    trmm_row_major_Left_Lower_NonTrans_template, "f32"
)
exo_strmm_row_major_Left_Lower_NonTrans = rename(
    exo_strmm_row_major_Left_Lower_NonTrans,
    exo_strmm_row_major_Left_Lower_NonTrans.name() + "",
)
exo_strmm_row_major_Left_Upper_Trans = specialize_trmm(
    trmm_row_major_Left_Upper_Trans_template, "f32"
)
exo_strmm_row_major_Left_Upper_Trans = rename(
    exo_strmm_row_major_Left_Upper_Trans,
    exo_strmm_row_major_Left_Upper_Trans.name() + "",
)
exo_strmm_row_major_Left_Lower_Trans = specialize_trmm(
    trmm_row_major_Left_Lower_Trans_template, "f32"
)
exo_strmm_row_major_Left_Lower_Trans = rename(
    exo_strmm_row_major_Left_Lower_Trans,
    exo_strmm_row_major_Left_Lower_Trans.name() + "",
)

f32_instructions = [
    C.Machine.load_instr_f32,
    C.Machine.store_instr_f32,
    C.Machine.mul_instr_f32,
    C.Machine.fmadd_instr_f32,
    C.Machine.broadcast_instr_f32,
    C.Machine.broadcast_scalar_instr_f32,
]

#################################################
# Generate specialized kernels for f64 precision
#################################################

entry_points = [
    exo_strmm_row_major_Left_Upper_NonTrans,
    # exo_dtrmm_row_major_Left_Upper_NonTrans,
    exo_strmm_row_major_Left_Lower_NonTrans,
    # exo_dtrmm_row_major_Lower_NonTrans,
    exo_strmm_row_major_Left_Upper_Trans,
    # exo_dtrmm_row_major_Upper_Trans,
    exo_strmm_row_major_Left_Lower_Trans,
    # exo_dtrmm_row_major_Lower_Trans,
]

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
