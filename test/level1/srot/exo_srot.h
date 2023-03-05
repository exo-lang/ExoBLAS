#pragma once

#include "exo_rot.h"

void exo_srot(const int N, float *X, const int incX,
                float *Y, const int incY, const float c, const float s) {
    if (incX == 1 && incY == 1) {
        exo_srot_stride_1(nullptr, N, 
            exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_1f32{.data = Y, .strides = {incY}},
            &c, &s);
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        if (incY < 0) {
            Y = Y + (1 - N) * incY;
        }
        exo_srot_stride_any(nullptr, N, 
            exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_1f32{.data = Y, .strides = {incY}},
            &c, &s);
    }
}
