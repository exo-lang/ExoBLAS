#pragma once

#include "exo_dot.h"

#define exo_dot(type, prefix, exo_type)                                     \
  type exo_##prefix##dot(const int N, const type *X, const int incX,        \
                         const type *Y, const int incY) {                   \
    if (incX == 1 && incY == 1) {                                           \
      type result;                                                          \
      exo_##prefix##dot_stride_1(                                           \
          nullptr, N, exo_win_1##exo_type##c{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type##c{.data = Y, .strides = {incY}}, &result);   \
      return result;                                                        \
    } else {                                                                \
      if (incX < 0) {                                                       \
        X = X + (1 - N) * incX;                                             \
      }                                                                     \
      if (incY < 0) {                                                       \
        Y = Y + (1 - N) * incY;                                             \
      }                                                                     \
      type result;                                                          \
      exo_##prefix##dot_stride_any(                                         \
          nullptr, N, exo_win_1##exo_type##c{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type##c{.data = Y, .strides = {incY}}, &result);   \
      return result;                                                        \
    }                                                                       \
  }

exo_dot(float, s, f32);
exo_dot(double, d, f64);
