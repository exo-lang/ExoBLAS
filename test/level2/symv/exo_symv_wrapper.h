#pragma once

#include <cblas.h>

#include "exo_symv.h"

#define exo_symv(type, prefix, exo_type)                                     \
  void exo_##prefix##symv(                                                   \
      const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N, \
      const type alpha, const type *A, const int lda, const type *X,         \
      const int incX, const type beta, type *Y, const int incY) {            \
    if (order != CBLAS_ORDER::CblasRowMajor) {                               \
      throw "Unsupported Exception";                                         \
    }                                                                        \
    if (incX < 0) {                                                          \
      X = X + (1 - N) * incX;                                                \
    }                                                                        \
    if (incY < 0) {                                                          \
      Y = Y + (1 - N) * incY;                                                \
    }                                                                        \
    if (incX == 1 && incY == 1) {                                            \
      exo_##prefix##symv_rm_stride_1(                                        \
          nullptr, Uplo, N, &alpha,                                          \
          exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},            \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}}, &beta,       \
          exo_win_1##exo_type{.data = Y, .strides = {incY}});                \
    } else {                                                                 \
      exo_##prefix##symv_rm_stride_any(                                      \
          nullptr, Uplo, N, &alpha,                                          \
          exo_win_2##exo_type##c{.data = A, .strides = {lda, 1}},            \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}}, &beta,       \
          exo_win_1##exo_type{.data = Y, .strides = {incY}});                \
    }                                                                        \
  }

exo_symv(float, s, f32);
exo_symv(double, d, f64);
