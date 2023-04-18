#pragma once

#include "exo_gbmv.h"

void exo_dgbmv(const int M, const int N, const int KL, const int KU,
               const double alpha, const double *A, const int lda,
               const double *X, const int incX, const double beta, double *Y,
               const int incY) {
  if (incX == 1 && incY == 1) {
    exo_dgbmv_row_major_NonTrans_stride_1(
        nullptr, M, N, KL, KU, &alpha, &beta,
        exo_win_2f64c{.data = A, .strides = {lda, 1}},
        exo_win_1f64c{.data = X, .strides = {incX}},
        exo_win_1f64{.data = Y, .strides = {incY}});
  } else {
    exo_dgbmv_row_major_NonTrans_stride_any(
        nullptr, M, N, KL, KU, &alpha, &beta,
        exo_win_2f64c{.data = A, .strides = {lda, 1}},
        exo_win_1f64c{.data = X, .strides = {incX}},
        exo_win_1f64{.data = Y, .strides = {incY}});
  }
}
