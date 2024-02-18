#pragma once

#include <math.h>

#include "exo_scal.h"

#define exo_scal(type, prefix, exo_type)                                    \
  void exo_##prefix##scal(const int N, const type alpha, type *X,           \
                          const int incX) {                                 \
    if (alpha == 1.0) {                                                     \
      return;                                                               \
    }                                                                       \
    if (incX == 1) {                                                        \
      if (alpha == 0.0) {                                                   \
        exo_##prefix##scal_alpha_0_stride_1(                                \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}); \
      } else {                                                              \
        exo_##prefix##scal_stride_1(                                        \
            nullptr, N, &alpha,                                             \
            exo_win_1##exo_type{.data = X, .strides = {incX}});             \
      }                                                                     \
    } else {                                                                \
      if (incX < 0) {                                                       \
        return;                                                             \
      }                                                                     \
      if (alpha == 0.0) {                                                   \
        exo_##prefix##scal_alpha_0_stride_any(                              \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}); \
      } else {                                                              \
        exo_##prefix##scal_stride_any(                                      \
            nullptr, N, &alpha,                                             \
            exo_win_1##exo_type{.data = X, .strides = {incX}});             \
      }                                                                     \
    }                                                                       \
  }

exo_scal(float, s, f32);
exo_scal(double, d, f64);
