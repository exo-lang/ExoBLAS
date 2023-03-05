#pragma once

#include <math.h>

#include "exo_nrm2.h"

float exo_snrm2(const int N, const float *X, const int incX) {
    if (incX == 1) {
        float result;
        exo_snrm2_stride_1(nullptr, N, 
            exo_win_1f32c{.data = X, .strides = {incX}},
            &result);
        return sqrtf(result);
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        float result;
        exo_snrm2_stride_any(nullptr, N, 
            exo_win_1f32c{.data = X, .strides = {incX}},
            &result);
        return sqrtf(result);
    }
}
