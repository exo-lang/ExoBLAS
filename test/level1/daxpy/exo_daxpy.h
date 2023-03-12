#pragma once

#include "exo_axpy.h"

void exo_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY) {
    if (alpha == 0.0) {
        return;
    }
    if (incX == 1 && incY == 1) {
        if (alpha == 1.0) {
            exo_daxpy_alpha_1_stride_1(nullptr, N, 
                exo_win_1f64c{.data = X, .strides = {incX}},
                exo_win_1f64{.data = Y, .strides = {incY}});
        } else {
            exo_daxpy_stride_1(nullptr, N, &alpha, 
                exo_win_1f64c{.data = X, .strides = {incX}},
                exo_win_1f64{.data = Y, .strides = {incY}});
        }
    } else {
        if (incX < 0) {
            X = X + (1 - N) * incX;
        }
        if (incY < 0) {
            Y = Y + (1 - N) * incY;
        }
        exo_daxpy_stride_any(nullptr, N, &alpha,
            exo_win_1f64c{.data = X, .strides = {incX}},
            exo_win_1f64{.data = Y, .strides = {incY}});
    }
}
