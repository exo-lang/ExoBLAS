#pragma once

#include "exo_swap.h"

#define exo_swap(type, prefix, exo_type)                                 \
  void exo_##prefix##swap(const int N, type *X, const int incX, type *Y, \
                          const int incY) {                              \
    if (incX == 1 && incY == 1) {                                        \
      exo_##prefix##swap_stride_1(                                       \
          nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type{.data = Y, .strides = {incY}});            \
    } else {                                                             \
      if (incX < 0) {                                                    \
        X = X + (1 - N) * incX;                                          \
      }                                                                  \
      if (incY < 0) {                                                    \
        Y = Y + (1 - N) * incY;                                          \
      }                                                                  \
      exo_##prefix##swap_stride_any(                                     \
          nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type{.data = Y, .strides = {incY}});            \
    }                                                                    \
  }

exo_swap(float, s, f32);
exo_swap(double, d, f64);
