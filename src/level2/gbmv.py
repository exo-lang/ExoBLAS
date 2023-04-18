from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *

from dot import exo_sdot_stride_1, dot_template, exo_ddot_stride_1
import exo_blas_config as C
from composed_schedules import (
    vectorize,
    interleave_execution,
    parallelize_reduction,
    interleave_outer_loop_with_inner_loop,
    apply_to_block,
    hoist_stmt,
    stage_expr,
)


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
                # exo_sdot_stride_1(kl+ku+1, a[i, 0:kl+ku+1], x[i-kl:i+ku+1], result)
            else:
                if kl + n - i > 0:
                    result = 0.0
                    for j in seq(0, kl + n - i):
                        result += a[i, j] * x[i + (j - kl)]
                    # exo_sdot_stride_1(kl+n-i, a[i, 0:kl+n-i], x[i-kl:n], result)
        else:
            if i + ku < n:
                result = 0.0
                for j in seq(0, i + ku + 1):
                    result += a[i, kl - i + j] * x[j]
                # exo_sdot_stride_1(i+ku+1, a[i, kl-i:ku+kl+1], x[0:i+ku+1], result)
            else:
                result = 0.0
                for j in seq(0, n):
                    result += a[i, kl - i + j] * x[j]
                # exo_sdot_stride_1(n, a[i, kl-i:kl-i+n], x[0:n], result)

        y[i] = beta * y[i] + alpha * result


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

    print("End:\n", stride_1)

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

if __name__ == "__main__":
    for p in entry_points:
        print(p)

__all__ = [p.name() for p in entry_points]
