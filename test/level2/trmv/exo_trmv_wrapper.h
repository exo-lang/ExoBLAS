#pragma once

#include <cblas.h>

#include "exo_trmv.h"

#define exo_trmv(type, prefix, exo_type)                                    \
  void exo_##prefix##trmv(                                                  \
      const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,             \
      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,        \
      const int N, const type *A, const int lda, type *X, const int incX) { \
    if (order != CBLAS_ORDER::CblasRowMajor) {                              \
      throw "Unsupported Exception";                                        \
    }                                                                       \
    if (incX < 0) {                                                         \
      X = X + (1 - N) * incX;                                               \
    }                                                                       \
    if (incX == 1) {                                                        \
      exo_##prefix##trmv_rm_stride_1(                                       \
          nullptr, Uplo, TransA, Diag, N,                                   \
          exo_win_1##exo_type{.data = X, .strides = {incX}},                \
          exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}});          \
    } else {                                                                \
      exo_##prefix##trmv_rm_stride_any(                                     \
          nullptr, Uplo, TransA, Diag, N,                                   \
          exo_win_1##exo_type{.data = X, .strides = {incX}},                \
          exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}});          \
    }                                                                       \
  }

exo_trmv(float, s, f32);
exo_trmv(double, d, f64);
