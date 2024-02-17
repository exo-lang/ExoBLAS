#pragma once

#include "exo_asum.h"

#define exo_asum(type, prefix, exo_type)                                    \
  type exo_##prefix##asum(const int N, const type *X, const int incX) {     \
    if (incX == 1) {                                                        \
      type result;                                                          \
      exo_##prefix##asum_stride_1(                                          \
          nullptr, N, exo_win_1##exo_type##c{.data = X, .strides = {incX}}, \
          &result);                                                         \
      return result;                                                        \
    } else {                                                                \
      if (incX < 0) {                                                       \
        return 0.0;                                                         \
      }                                                                     \
      type result;                                                          \
      exo_##prefix##asum_stride_any(                                        \
          nullptr, N, exo_win_1##exo_type##c{.data = X, .strides = {incX}}, \
          &result);                                                         \
      return result;                                                        \
    }                                                                       \
  }

exo_asum(float, s, f32);
exo_asum(double, d, f64);
