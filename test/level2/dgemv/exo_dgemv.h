#pragma once

#include <cblas.h>

#include "exo_gemv.h"

void exo_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
               const int M, const int N, const double alpha, const double *A,
               const int lda, const double *X, const int incX,
               const double beta, double *Y, const int incY) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }

  if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
    if (incX == 1 && incY == 1) {
      exo_dgemv_rm_nt_stride_1(nullptr, M, N, &alpha, &beta,
                               exo_win_2f64c{.data = A, .strides = {lda, 1}},
                               exo_win_1f64c{.data = X, .strides = {incX}},
                               exo_win_1f64{.data = Y, .strides = {incY}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      if (incY < 0) {
        Y = Y + (1 - M) * incY;
      }
      exo_dgemv_rm_nt_stride_any(nullptr, M, N, &alpha, &beta,
                                 exo_win_2f64c{.data = A, .strides = {lda, 1}},
                                 exo_win_1f64c{.data = X, .strides = {incX}},
                                 exo_win_1f64{.data = Y, .strides = {incY}});
    }
  } else {
    if (incX == 1 && incY == 1) {
      exo_dgemv_rm_t_stride_1(nullptr, N, M, &alpha, &beta,
                              exo_win_2f64c{.data = A, .strides = {lda, 1}},
                              exo_win_1f64c{.data = X, .strides = {incX}},
                              exo_win_1f64{.data = Y, .strides = {incY}});
    } else {
      if (incX < 0) {
        X = X + (1 - M) * incX;
      }
      if (incY < 0) {
        Y = Y + (1 - N) * incY;
      }
      exo_dgemv_rm_t_stride_any(nullptr, N, M, &alpha, &beta,
                                exo_win_2f64c{.data = A, .strides = {lda, 1}},
                                exo_win_1f64c{.data = X, .strides = {incX}},
                                exo_win_1f64{.data = Y, .strides = {incY}});
    }
  }
}
