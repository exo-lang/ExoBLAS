#pragma once

#include "exo_gbmv.h"

void exo_sgbmv(const int M, const int N, const int KL, const int KU,
               const float alpha, const float *A, const int lda, const float *X,
               const int incX, const float beta, float *Y, const int incY) {
  if (incX == 1 && incY == 1) {
    exo_sgbmv_row_major_NonTrans_stride_1(
        nullptr, M, N, KL, KU, alpha, beta,
        exo_win_2f32c{.data = A, .strides = {lda, 1}},
        exo_win_1f32c{.data = X, .strides = {incX}},
        exo_win_1f32{.data = Y, .strides = {incY}});
  } else {
    exo_sgbmv_row_major_NonTrans_stride_any(
        nullptr, M, N, KL, KU, alpha, beta,
        exo_win_2f32c{.data = A, .strides = {lda, 1}},
        exo_win_1f32c{.data = X, .strides = {incX}},
        exo_win_1f32{.data = Y, .strides = {incY}});
  }
}
