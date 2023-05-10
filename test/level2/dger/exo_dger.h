#pragma once

#include "exo_ger.h"

void exo_dger(const int M, const int N, const double alpha, const double *X,
              const int incX, const double *Y, const int incY, double *A,
              const int lda) {
  if (alpha == 0.0) {
    return;
  }
  if (incX == 1 && incY == 1) {
    exo_dger_row_major_stride_1(nullptr, M, N, alpha,
                                exo_win_1f64c{.data = X, .strides = {incX}},
                                exo_win_1f64c{.data = Y, .strides = {incY}},
                                exo_win_2f64{.data = A, .strides = {lda, 1}});
  } else {
    if (incX < 0) {
      X = X + (1 - M) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    exo_dger_row_major_stride_any(nullptr, M, N, alpha,
                                  exo_win_1f64c{.data = X, .strides = {incX}},
                                  exo_win_1f64c{.data = Y, .strides = {incY}},
                                  exo_win_2f64{.data = A, .strides = {lda, 1}});
  }
}
