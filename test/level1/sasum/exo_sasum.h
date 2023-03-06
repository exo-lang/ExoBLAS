#pragma once

#include <math.h>

#include "exo_asum.h"

float exo_sasum(const int N, const float *X, const int incX) {
    if (incX == 1) {
        float result;
        exo_sasum_stride_1(nullptr, N, 
            exo_win_1f32c{.data = X, .strides = {incX}},
            &result);
        return result;
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        float result;
        exo_sasum_stride_any(nullptr, N, 
            exo_win_1f32c{.data = X, .strides = {incX}},
            &result);
        return result;
    }
}
