#pragma once

#include "exo_dsdot.h"

double exo_dsdot(const int N, const float *X, const int incX, const float *Y,
                 const int incY) {
  if (incX == 1 && incY == 1) {
    double result;
    exo_dsdot_stride_1(nullptr, N, exo_win_1f32c{.data = X, .strides = {incX}},
                       exo_win_1f32c{.data = Y, .strides = {incY}}, &result);
    return result;
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    double result;
    exo_dsdot_stride_any(nullptr, N,
                         exo_win_1f32c{.data = X, .strides = {incX}},
                         exo_win_1f32c{.data = Y, .strides = {incY}}, &result);
    return result;
  }
}
