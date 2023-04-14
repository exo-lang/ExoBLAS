#pragma once

#include <cblas.h>

#include "exo_gemv.h"

void exo_sgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
               const int M, const int N, const float alpha, const float *A,
               const int lda, const float *X, const int incX, const float beta,
               float *Y, const int incY) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }

  if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
    if (incX == 1 && incY == 1) {
      exo_sgemv_row_major_NonTrans_stride_1(
          nullptr, M, N, &alpha, &beta,
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      if (incY < 0) {
        Y = Y + (1 - M) * incY;
      }
      exo_sgemv_row_major_NonTrans_stride_any(
          nullptr, M, N, &alpha, &beta,
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    }
  } else {
    if (incX == 1 && incY == 1) {
      exo_sgemv_row_major_Trans_stride_1(
          nullptr, M, N, &alpha, &beta,
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      if (incX < 0) {
        X = X + (1 - M) * incX;
      }
      if (incY < 0) {
        Y = Y + (1 - N) * incY;
      }
      exo_sgemv_row_major_Trans_stride_any(
          nullptr, M, N, &alpha, &beta,
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    }
  }
}
