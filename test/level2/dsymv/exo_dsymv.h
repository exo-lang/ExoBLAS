#pragma once

#include <cblas.h>

#include "exo_symv.h"

void exo_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const double alpha, const double *A, const int lda,
               const double *X, const int incX, const double beta, double *Y,
               const int incY) {
  if (order != CBLAS_ORDER::CblasRowMajor) {
    throw "Unsupported Exception";
  }
  if (incX < 0) {
    X = X + (1 - N) * incX;
  }
  if (incY < 0) {
    Y = Y + (1 - N) * incY;
  }
  if (Uplo == CBLAS_UPLO::CblasUpper) {
    if (incX == 1 && incY == 1) {
      exo_dsymv_rm_u_stride_1(
          nullptr, N, &alpha, exo_win_2f64c{.data = A, .strides = {lda, 1}},
          exo_win_1f64c{.data = X, .strides = {incX}}, &beta,
          exo_win_1f64{.data = Y, .strides = {incY}});
    } else {
      exo_dsymv_rm_u_stride_any(
          nullptr, N, &alpha, exo_win_2f64c{.data = A, .strides = {lda, 1}},
          exo_win_1f64c{.data = X, .strides = {incX}}, &beta,
          exo_win_1f64{.data = Y, .strides = {incY}});
    }
  } else {
    if (incX == 1 && incY == 1) {
      exo_dsymv_rm_l_stride_1(
          nullptr, N, &alpha, exo_win_2f64c{.data = A, .strides = {lda, 1}},
          exo_win_1f64c{.data = X, .strides = {incX}}, &beta,
          exo_win_1f64{.data = Y, .strides = {incY}});
    } else {
      exo_dsymv_rm_l_stride_any(
          nullptr, N, &alpha, exo_win_2f64c{.data = A, .strides = {lda, 1}},
          exo_win_1f64c{.data = X, .strides = {incX}}, &beta,
          exo_win_1f64{.data = Y, .strides = {incY}});
    }
  }
}
