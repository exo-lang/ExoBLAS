#pragma once

#include "exo_syr.h"

void exo_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
              const int N, const float alpha, const float *X, const int incX,
              float *A, const int lda) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (alpha == 0.0) {
    return;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (incX == 1) {
      exo_ssyr_row_major_Upper_stride_1(
          nullptr, N, alpha, exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_2f32{.data = A, .strides = {lda, 1}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      exo_ssyr_row_major_Upper_stride_any(
          nullptr, N, alpha, exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_2f32{.data = A, .strides = {lda, 1}});
    }
  } else {
    if (incX == 1) {
      exo_ssyr_row_major_Lower_stride_1(
          nullptr, N, alpha, exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_2f32{.data = A, .strides = {lda, 1}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      exo_ssyr_row_major_Lower_stride_any(
          nullptr, N, alpha, exo_win_1f32c{.data = X, .strides = {incX}},
          exo_win_2f32{.data = A, .strides = {lda, 1}});
    }
  }
}
