#pragma once

#include "exo_asum.h"

double exo_dasum(const int N, const double *X, const int incX) {
    if (incX == 1) {
        double result;
        exo_dasum_stride_1(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            &result);
        return result;
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        double result;
        exo_dasum_stride_any(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            &result);
        return result;
    }
}
