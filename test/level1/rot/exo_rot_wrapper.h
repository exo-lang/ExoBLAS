#pragma once

#include "exo_rot.h"

#define exo_rot(type, prefix, exo_type)                                  \
  void exo_##prefix##rot(const int N, type *X, const int incX, type *Y,  \
                         const int incY, const type c, const type s) {   \
    if (incX == 1 && incY == 1) {                                        \
      exo_##prefix##rot_stride_1(                                        \
          nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type{.data = Y, .strides = {incY}}, &c, &s);    \
    } else {                                                             \
      if (incX < 0) {                                                    \
        X = X + (1 - N) * incX;                                          \
      }                                                                  \
      if (incY < 0) {                                                    \
        Y = Y + (1 - N) * incY;                                          \
      }                                                                  \
      exo_##prefix##rot_stride_any(                                      \
          nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
          exo_win_1##exo_type{.data = Y, .strides = {incY}}, &c, &s);    \
    }                                                                    \
  }

exo_rot(float, s, f32);
exo_rot(double, d, f64);
