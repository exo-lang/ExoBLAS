#pragma once

#include "exo_copy.h"

void exo_dcopy(const int N, const double *X, const int incX, double *Y,
               const int incY) {
  if (incX == 1 && incY == 1) {
    exo_dcopy_stride_1(nullptr, N, exo_win_1f64c{.data = X, .strides = {incX}},
                       exo_win_1f64{.data = Y, .strides = {incY}});
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    exo_dcopy_stride_any(nullptr, N,
                         exo_win_1f64c{.data = X, .strides = {incX}},
                         exo_win_1f64{.data = Y, .strides = {incY}});
  }
}
