#pragma once

#include <cblas.h>

#include "exo_tpmv.h"

void exo_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const float *A, float *X,
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
          exo_stpmv_row_major_Upper_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        } else {
          exo_stpmv_row_major_Upper_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        }
      } else {
        if (incX == 1) {
          exo_stpmv_row_major_Upper_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        } else {
          exo_stpmv_row_major_Upper_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        }
      }
    } else {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
        exo_stpmv_row_major_Upper_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stpmv_row_major_Upper_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      }
      } else {
        if (incX == 1) {
        exo_stpmv_row_major_Upper_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stpmv_row_major_Upper_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      }
      }
    }
  } else {
    if (TransA == CBLAS_TRANSPOSE::CblasNoTrans) {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
          exo_stpmv_row_major_Lower_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        } else {
          exo_stpmv_row_major_Lower_NonTrans_Unit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        }
      } else {
        if (incX == 1) {
          exo_stpmv_row_major_Lower_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        } else {
          exo_stpmv_row_major_Lower_NonTrans_NonUnit_stride_any(
              nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
              A);
        }
      }
    } else {
      if (Diag == CBLAS_DIAG::CblasUnit) {
        if (incX == 1) {
        exo_stpmv_row_major_Lower_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stpmv_row_major_Lower_Trans_Unit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      }
      } else {
        if (incX == 1) {
        exo_stpmv_row_major_Lower_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      } else {
        exo_stpmv_row_major_Lower_Trans_NonUnit_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          A,
          Diag == CBLAS_DIAG::CblasUnit);
      }
      }
    }
  }
}
