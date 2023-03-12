#pragma once

#include <math.h>

#include "exo_scal.h"

void exo_dscal(const int N, const double alpha, double *X, const int incX) {
    if (alpha == 1.0) {
        return;
    }
    if (incX == 1) {
        if (alpha == 0.0) {
            exo_dscal_alpha_0_stride_1(nullptr, N,
                exo_win_1f64{.data = X, .strides = {incX}});
        } else {
            exo_dscal_stride_1(nullptr, N, &alpha,
                exo_win_1f64{.data = X, .strides = {incX}});
        }
    } else {
        if (incX < 0) {
            return;
        }
        if (alpha == 0.0) {
            exo_dscal_alpha_0_stride_any(nullptr, N,
                exo_win_1f64{.data = X, .strides = {incX}});
        } else {
            exo_dscal_stride_any(nullptr, N, &alpha,
                exo_win_1f64{.data = X, .strides = {incX}});
        }
    }
}
