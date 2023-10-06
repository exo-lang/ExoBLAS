#pragma once

#include "exo_rotm.h"

void exo_srotm(const int N, float *X, const int incX, float *Y, const int incY,
               const float *P) {
  float H[4] = {P[1], P[3], P[2], P[4]};
  if (incX == 1 && incY == 1) {
    if (P[0] == -1.0f) {
      exo_srotm_flag_neg_one_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 0.0f) {
      exo_srotm_flag_zero_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 1.0f) {
      exo_srotm_flag_one_stride_1(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    }
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    if (P[0] == -1.0f) {
      exo_srotm_flag_neg_one_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 0.0f) {
      exo_srotm_flag_zero_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 1.0f) {
      exo_srotm_flag_one_stride_any(
          nullptr, N, exo_win_1f32{.data = X, .strides = {incX}},
          exo_win_1f32{.data = Y, .strides = {incY}}, H);
    }
  }
}
