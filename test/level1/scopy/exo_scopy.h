#pragma once

#include "exo_copy.h"

void exo_scopy(const int N, const float *X, const int incX, float *Y,
               const int incY) {
  if (incX == 1 && incY == 1) {
    exo_scopy_stride_1(nullptr, N, exo_win_1f32c{.data = X, .strides = {incX}},
                       exo_win_1f32{.data = Y, .strides = {incY}});
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    exo_scopy_stride_any(nullptr, N,
                         exo_win_1f32c{.data = X, .strides = {incX}},
                         exo_win_1f32{.data = Y, .strides = {incY}});
  }
}
