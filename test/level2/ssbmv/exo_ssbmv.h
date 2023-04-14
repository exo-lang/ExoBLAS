#pragma once

#include <cblas.h>

#include "exo_sbmv.h"

void exo_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const int K, const float alpha, const float *A,
               const int lda, const float *X, const int incX, const float beta,
               float *Y, const int incY) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (incX < 0) {
    X = X + (1 - N) * incX;
  }
  if (incY < 0) {
    Y = Y + (1 - N) * incY;
  }
  if (beta != 1.0) {
    if (incY == 1) {
      exo_ssbmv_scal_y_stride_1(nullptr, N, &beta,
                                exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      exo_ssbmv_scal_y_stride_any(nullptr, N, &beta,
                                  exo_win_1f32{.data = Y, .strides = {incY}});
    }
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (incX == 1 && incY == 1) {
      exo_ssbmv_row_major_Upper_stride_1(
          nullptr, N, K, &alpha, exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      exo_ssbmv_row_major_Upper_stride_any(
          nullptr, N, K, &alpha, exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    }
  } else {
    if (incX == 1 && incY == 1) {
      exo_ssbmv_row_major_Lower_stride_1(
          nullptr, N, K, &alpha, exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    } else {
      exo_ssbmv_row_major_Lower_stride_any(
          nullptr, N, K, &alpha, exo_win_2f32c{.data = A, .strides = {lda, 1}},
          exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}});
    }
  }
}
