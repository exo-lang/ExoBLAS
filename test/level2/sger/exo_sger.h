#pragma once

#include "exo_ger.h"

void exo_sger(const int M, const int N, const float alpha, const float *X,
              const int incX, const float *Y, const int incY, float *A,
              const int lda) {
  if (alpha == 0.0) {
    return;
  }
  if (incX == 1 && incY == 1) {
    exo_sger_rm_stride_1(nullptr, M, N, &alpha,
                         exo_win_1f32c{.data = X, .strides = {incX}},
                         exo_win_1f32c{.data = Y, .strides = {incY}},
                         exo_win_2f32{.data = A, .strides = {lda, 1}});
  } else {
    if (incX < 0) {
      X = X + (1 - M) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    exo_sger_rm_stride_any(nullptr, M, N, &alpha,
                           exo_win_1f32c{.data = X, .strides = {incX}},
                           exo_win_1f32c{.data = Y, .strides = {incY}},
                           exo_win_2f32{.data = A, .strides = {lda, 1}});
  }
}
