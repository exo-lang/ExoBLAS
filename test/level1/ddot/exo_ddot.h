#pragma once

#include "exo_dot.h"

double exo_ddot(const int N, const double  *X, const int incX,
                  const double  *Y, const int incY) {
    if (incX == 1 && incY == 1) {
        double result;
        exo_ddot_stride_1(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            exo_win_1f64c{.data = Y, .strides = {incY}},
            &result);
        return result;
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        if (incY < 0) {
            Y = Y + (1 - N) * incY;
        }
        double result;
        exo_ddot_stride_any(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            exo_win_1f64c{.data = Y, .strides = {incY}},
            &result);
        return result;
    }
}
