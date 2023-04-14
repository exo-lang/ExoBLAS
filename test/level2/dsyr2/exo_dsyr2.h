#pragma once

#include "exo_syr2.h"

void exo_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const double alpha, const double *X, const int incX,
               const double *Y, const int incY, double *A, const int lda) {
  if (alpha == 0.0) {
    return;
  }
  if (incX < 0) {
    X = X + (1 - N) * incX;
  }
  if (incY < 0) {
    Y = Y + (1 - N) * incY;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (incX == 1 && incY == 1) {
      exo_dsyr2_row_major_Upper_stride_1(
          nullptr, N, &alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_1f64c{.data = Y, .strides = {incY}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    } else {
      exo_dsyr2_row_major_Upper_stride_any(
          nullptr, N, &alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_1f64c{.data = Y, .strides = {incY}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    }
  } else {
    if (incX == 1 && incY == 1) {
      exo_dsyr2_row_major_Lower_stride_1(
          nullptr, N, &alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_1f64c{.data = Y, .strides = {incY}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    } else {
      exo_dsyr2_row_major_Lower_stride_any(
          nullptr, N, &alpha, exo_win_1f64c{.data = X, .strides = {incX}},
          exo_win_1f64c{.data = Y, .strides = {incY}},
          exo_win_2f64{.data = A, .strides = {lda, 1}});
    }
  }
}
