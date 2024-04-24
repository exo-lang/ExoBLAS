#pragma once

#include "exo_syr.h"

#define exo_syr(type, prefix, exo_type)                                   \
  void exo_##prefix##syr(const enum CBLAS_ORDER order,                    \
                         const enum CBLAS_UPLO Uplo, const int N,         \
                         const type alpha, const type *X, const int incX, \
                         type *A, const int lda) {                        \
    if (order != CBLAS_ORDER::CblasRowMajor) {                            \
      throw "Unsupported Exception";                                      \
    }                                                                     \
    if (alpha == 0.0) {                                                   \
      return;                                                             \
    }                                                                     \
    if (incX == 1) {                                                      \
      exo_##prefix##syr_rm_stride_1(                                      \
          nullptr, Uplo, N, &alpha,                                       \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},           \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},           \
          exo_win_2##exo_type{.data = A, .strides = {lda, 1}});           \
    } else {                                                              \
      if (incX < 0) {                                                     \
        X = X + (1 - N) * incX;                                           \
      }                                                                   \
      exo_##prefix##syr_rm_stride_any(                                    \
          nullptr, Uplo, N, &alpha,                                       \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},           \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},           \
          exo_win_2##exo_type{.data = A, .strides = {lda, 1}});           \
    }                                                                     \
  }

exo_syr(float, s, f32);
exo_syr(double, d, f64);
