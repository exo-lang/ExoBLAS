#pragma once

#include "exo_axpy.h"

#define exo_axpy(type, prefix, exo_type)                                      \
  void exo_##prefix##axpy(const int N, const type alpha, const type *X,       \
                          const int incX, type *Y, const int incY) {          \
    if (alpha == 0.0) {                                                       \
      return;                                                                 \
    }                                                                         \
    if (incX == 1 && incY == 1) {                                             \
      if (alpha == 1.0) {                                                     \
        exo_##prefix##axpy_alpha_1_stride_1(                                  \
            nullptr, N, exo_win_1##exo_type##c{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});               \
      } else {                                                                \
        exo_##prefix##axpy_stride_1(                                          \
            nullptr, N, &alpha,                                               \
            exo_win_1##exo_type##c{.data = X, .strides = {incX}},             \
            exo_win_1##exo_type{.data = Y, .strides = {incY}});               \
      }                                                                       \
    } else {                                                                  \
      if (incX < 0) {                                                         \
        X = X + (1 - N) * incX;                                               \
      }                                                                       \
      if (incY < 0) {                                                         \
        Y = Y + (1 - N) * incY;                                               \
      }                                                                       \
      exo_##prefix##axpy_stride_any(                                          \
          nullptr, N, &alpha,                                                 \
          exo_win_1##exo_type##c{.data = X, .strides = {incX}},               \
          exo_win_1##exo_type{.data = Y, .strides = {incY}});                 \
    }                                                                         \
  }

exo_axpy(float, s, f32);
exo_axpy(double, d, f64);
