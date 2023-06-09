#pragma once

#include <math.h>

#include "exo_scal.h"

void exo_sscal(const int N, const float alpha, float *X, const int incX) {
  if (alpha == 1.0f) {
    return;
  }
  if (incX == 1) {
    if (alpha == 0.0f) {
      exo_sscal_alpha_0_stride_1(nullptr, N,
                                 exo_win_1f32{.data = X, .strides = {incX}});
    } else {
      exo_sscal_stride_1(nullptr, N, alpha,
                         exo_win_1f32{.data = X, .strides = {incX}});
    }
  } else {
    if (incX < 0) {
      return;
    }
    if (alpha == 0.0f) {
      exo_sscal_alpha_0_stride_any(nullptr, N,
                                   exo_win_1f32{.data = X, .strides = {incX}});
    } else {
      exo_sscal_stride_any(nullptr, N, alpha,
                           exo_win_1f32{.data = X, .strides = {incX}});
    }
  }
}
