#pragma once

#include <cblas.h>

#include "exo_ger.h"

#define exo_ger(type, prefix, exo_type)                                 \
  void exo_##prefix##ger(const enum CBLAS_ORDER order, const int M,     \
                         const int N, const type alpha, const type *X,  \
                         const int incX, const type *Y, const int incY, \
                         type *A, const int lda) {                      \
    if (alpha == 0.0) {                                                 \
      return;                                                           \
    }                                                                   \
    if (incX == 1 && incY == 1) {                                       \
      exo_##prefix##ger_rm_stride_1(                                    \
          nullptr, M, N, &alpha,                                        \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
          exo_win_1##exo_type##c{.data = Y, .strides = {incY}},         \
          exo_win_2##exo_type{.data = A, .strides = {lda, 1}});         \
    } else {                                                            \
      if (incX < 0) {                                                   \
        X = X + (1 - M) * incX;                                         \
      }                                                                 \
      if (incY < 0) {                                                   \
        Y = Y + (1 - N) * incY;                                         \
      }                                                                 \
      exo_##prefix##ger_rm_stride_any(                                  \
          nullptr, M, N, &alpha,                                        \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},         \
          exo_win_1##exo_type##c{.data = Y, .strides = {incY}},         \
          exo_win_2##exo_type{.data = A, .strides = {lda, 1}});         \
    }                                                                   \
  }

exo_ger(float, s, f32);
exo_ger(double, d, f64);
