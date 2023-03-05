#pragma once

#include <math.h>

#include "exo_scal.h"

void exo_sscal(const int N, const float alpha, float *X, const int incX) {
    if (incX == 1) {
        exo_sscal_stride_1(nullptr, N, &alpha,
            exo_win_1f32{.data = X, .strides = {incX}});
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        exo_sscal_stride_any(nullptr, N, &alpha,
            exo_win_1f32{.data = X, .strides = {incX}});
    }
}
