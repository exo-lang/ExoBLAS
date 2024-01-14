from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

from dot import exo_sdot_stride_1, dot_template, exo_ddot_stride_1
import exo_blas_config as C
from composed_schedules import (
    scalar_to_simd,
    interleave_execution,
    interleave_outer_loop_with_inner_loop,
    hoist_stmt,
    stage_expr,
)


### EXO_LOC ALGORITHM START ###
@proc
def gbmv_row_major_NonTrans(
    m: size,
    n: size,
    kl: size,
    ku: size,
    alpha: R,
    beta: R,
    a: [R][m, ku + kl + 1],
    x: [R][n],
    y: [R][m],
):
    assert kl + 1 <= m
    assert ku + 1 <= n

    for i in seq(0, m):
        result: R

        # j in [max(0, i - kl), min(n - 1, i + ku)]
        if i - kl >= 0:
            if i + ku < n:
                result = 0.0
                for j in seq(0, kl + ku + 1):
                    result += a[i, j] * x[i + (j - kl)]
            else:
                if kl + n - i > 0:
                    result = 0.0
                    for j in seq(0, kl + n - i):
                        result += a[i, j] * x[i + (j - kl)]
        else:
            if i + ku < n:
                result = 0.0
                for j in seq(0, i + ku + 1):
                    result += a[i, kl - i + j] * x[j]
            else:
                result = 0.0
                for j in seq(0, n):
                    result += a[i, kl - i + j] * x[j]

        y[i] = beta * y[i] + alpha * result


### EXO_LOC ALGORITHM END ###


### EXO_LOC SCHEDULE START ###
def specialize_sdot(precision):
    specialized = sdot_template

    for arg in ["x", "y", "result"]:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def specialize_gbmv(precision):
    prefix = "s" if precision == "f32" else "d"
    name = gbmv_row_major_NonTrans.name()
    specialized = rename(gbmv_row_major_NonTrans, "exo_" + prefix + name)

    for arg in ["alpha", "beta", "x", "y", "a", "result"]:
        specialized = set_precision(specialized, arg, precision)

    return specialized


def schedule_stride_1(precision):
    stride_1 = specialize_gbmv(precision)
    stride_1 = rename(stride_1, stride_1.name() + "_stride_1")
    stride_1 = stride_1.add_assertion("stride(x, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(y, 0) == 1")
    stride_1 = stride_1.add_assertion("stride(a, 1) == 1")

    scheduled_sdot = exo_sdot_stride_1 if precision == "f32" else exo_ddot_stride_1
    dot_template.unsafe_assert_eq(scheduled_sdot)

    for i in range(4):
        stride_1 = replace(stride_1, stride_1.find_loop("j").expand(1, 0), dot_template)
        stride_1 = call_eqv(stride_1, "dot_template", scheduled_sdot)

    return simplify(stride_1)


#################################################
# Generate specialized kernels for f32 precision
#################################################

exo_sgbmv_row_major_NonTrans_stride_any = specialize_gbmv("f32")
exo_sgbmv_row_major_NonTrans_stride_any = rename(
    exo_sgbmv_row_major_NonTrans_stride_any,
    exo_sgbmv_row_major_NonTrans_stride_any.name() + "_stride_any",
)

exo_sgbmv_row_major_NonTrans_stride_1 = schedule_stride_1("f32")

entry_points = [
    exo_sgbmv_row_major_NonTrans_stride_any,
    exo_sgbmv_row_major_NonTrans_stride_1,
]


#################################################
# Generate specialized kernels for f64 precision
#################################################

exo_dgbmv_row_major_NonTrans_stride_any = specialize_gbmv("f64")
exo_dgbmv_row_major_NonTrans_stride_any = rename(
    exo_dgbmv_row_major_NonTrans_stride_any,
    exo_dgbmv_row_major_NonTrans_stride_any.name() + "_stride_any",
)

exo_dgbmv_row_major_NonTrans_stride_1 = schedule_stride_1("f64")

entry_points = [
    exo_sgbmv_row_major_NonTrans_stride_any,
    exo_sgbmv_row_major_NonTrans_stride_1,
    exo_dgbmv_row_major_NonTrans_stride_any,
    exo_dgbmv_row_major_NonTrans_stride_1,
]
### EXO_LOC SCHEDULE END ###


if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
