#pragma once

#include <cblas.h>

#include "exo_trmv.h"

void exo_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const float *A, const int lda, float *X,
               const int incX) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (incX < 0) {
    X = X + (1 - N) * incX;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
          exo_strmv_row_major_Upper_NonTrans_Unit_stride_1(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        } else {
          exo_strmv_row_major_Upper_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        }
      } else {
        if (incX == 1) {
          exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_1(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        } else {
          exo_strmv_row_major_Upper_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        }
      }
    } else {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
        exo_strmv_row_major_Upper_Trans_Unit_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_strmv_row_major_Upper_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      }
      } else {
        if (incX == 1) {
        exo_strmv_row_major_Upper_Trans_NonUnit_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_strmv_row_major_Upper_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      }
      }
    }
  } else {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
          exo_strmv_row_major_Lower_NonTrans_Unit_stride_1(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        } else {
          exo_strmv_row_major_Lower_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        }
      } else {
        if (incX == 1) {
          exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_1(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        } else {
          exo_strmv_row_major_Lower_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              exo_win_2f32c{.data = A, .strides = {lda, 1}});
        }
      }
    } else {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
        exo_strmv_row_major_Lower_Trans_Unit_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_strmv_row_major_Lower_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      }
      } else {
        if (incX == 1) {
        exo_strmv_row_major_Lower_Trans_NonUnit_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_strmv_row_major_Lower_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_2f32c{.data = A, .strides = {lda, 1}},
          Diag == CBLAS_DIAG::CblasUnit);
      }
      }
    }
  }
}
