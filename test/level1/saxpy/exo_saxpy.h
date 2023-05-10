#pragma once

#include "exo_axpy.h"

void exo_saxpy(const int N, const float alpha, const float *X, const int incX,
               float *Y, const int incY) {
  if (alpha == 0.0f) {
    return;
  }
  if (incX == 1 && incY == 1) {
    if (alpha == 1.0f) {
      exo_saxpy_alpha_1_stride_1(nullptr, N,
                                 exo_win_1f32c{.data = X, .strides = {incX}},
                                 exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      exo_saxpy_stride_1(nullptr, N, alpha,
                         exo_win_1f32c{.data = X, .strides = {incX}},
                         exo_win_1f32{.data = Y, .strides = {incY}});
    }
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    exo_saxpy_stride_any(nullptr, N, alpha,
                         exo_win_1f32c{.data = X, .strides = {incX}},
                         exo_win_1f32{.data = Y, .strides = {incY}});
  }
}
