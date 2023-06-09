#pragma once

#include "exo_syr.h"

void exo_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
              const int N, const double alpha, const double *X, const int incX,
              double *A, const int lda) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (alpha == 0.0) {
    return;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (incX == 1) {
      exo_dsyr_row_major_Upper_stride_1(
          nullptr, N, alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      exo_dsyr_row_major_Upper_stride_any(
          nullptr, N, alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    }
  } else {
    if (incX == 1) {
      exo_dsyr_row_major_Lower_stride_1(
          nullptr, N, alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    } else {
      if (incX < 0) {
        X = X + (1 - N) * incX;
      }
      exo_dsyr_row_major_Lower_stride_any(
          nullptr, N, alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    }
  }
}
