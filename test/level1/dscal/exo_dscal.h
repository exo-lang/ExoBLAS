#pragma once

#include <math.h>

#include "exo_scal.h"

void exo_dscal(const int N, const double alpha, double *X, const int incX) {
    if (incX == 1) {
        exo_dscal_stride_1(nullptr, N, &alpha,
            exo_win_1f64{.data = X, .strides = {incX}});
    } else {
        if (incX < 0) {
            return;
        }
        exo_dscal_stride_any(nullptr, N, &alpha,
            exo_win_1f64{.data = X, .strides = {incX}});
    }
}
