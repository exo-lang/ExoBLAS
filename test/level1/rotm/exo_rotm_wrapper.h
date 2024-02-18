#pragma once

#include "exo_rotm.h"

#define exo_rotm(type, prefix, exo_type)                                   \
  void exo_##prefix##rotm(const int N, type *X, const int incX, type *Y,   \
                          const int incY, const type *P) {                 \
    type H[4] = {P[1], P[3], P[2], P[4]};                                  \
    if (incX == 1 && incY == 1) {                                          \
      if (P[0] == -1.0) {                                                  \
        exo_##prefix##rotm_flag_neg_one_stride_1(                          \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      } else if (P[0] == 0.0) {                                            \
        exo_##prefix##rotm_flag_zero_stride_1(                             \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      } else if (P[0] == 1.0) {                                            \
        exo_##prefix##rotm_flag_one_stride_1(                              \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      }                                                                    \
    } else {                                                               \
      if (incX < 0) {                                                      \
        X = X + (1 - N) * incX;                                            \
      }                                                                    \
      if (incY < 0) {                                                      \
        Y = Y + (1 - N) * incY;                                            \
      }                                                                    \
      if (P[0] == -1.0) {                                                  \
        exo_##prefix##rotm_flag_neg_one_stride_any(                        \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      } else if (P[0] == 0.0) {                                            \
        exo_##prefix##rotm_flag_zero_stride_any(                           \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      } else if (P[0] == 1.0) {                                            \
        exo_##prefix##rotm_flag_one_stride_any(                            \
            nullptr, N, exo_win_1##exo_type{.data = X, .strides = {incX}}, \
            exo_win_1##exo_type{.data = Y, .strides = {incY}}, H);         \
      }                                                                    \
    }                                                                      \
  }

exo_rotm(float, s, f32);
exo_rotm(double, d, f64);
