#pragma once

#include <cblas.h>

#include "exo_tbmv.h"

void exo_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const int K, const float *A, const int lda,
               float *X, const int incX) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (incX < 0) {
    X = X + (1 - N) * incX;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
      if (incX == 1) {
        exo_stbmv_row_major_Upper_NonTrans_stride_1(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stbmv_row_major_Upper_NonTrans_stride_any(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      }
    } else {
      if (incX == 1) {
        exo_stbmv_row_major_Upper_Trans_stride_1(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stbmv_row_major_Upper_Trans_stride_any(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      }
    }
  } else {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
      if (incX == 1) {
        exo_stbmv_row_major_Lower_NonTrans_stride_1(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stbmv_row_major_Lower_NonTrans_stride_any(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      }
    } else {
      if (incX == 1) {
        exo_stbmv_row_major_Lower_Trans_stride_1(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stbmv_row_major_Lower_Trans_stride_any(
            nullptr, N, K, exo_win_1f32{.data = X, .strides = {incX}},
            exo_win_2f32c{.data = A, .strides = {lda, 1}},
            Diag == CBLAS_DIAG::CblasUnit);
      }
    }
  }
}
