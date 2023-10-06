#pragma once

#include "exo_rotm.h"

void exo_drotm(const int N, double *X, const int incX, double *Y,
               const int incY, const double *P) {
  double H[4] = {P[1], P[3], P[2], P[4]};
  if (incX == 1 && incY == 1) {
    if (P[0] == -1.0) {
      exo_drotm_flag_neg_one_stride_1(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 0.0) {
      exo_drotm_flag_zero_stride_1(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 1.0) {
      exo_drotm_flag_one_stride_1(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    }
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    if (P[0] == -1.0) {
      exo_drotm_flag_neg_one_stride_any(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 0.0) {
      exo_drotm_flag_zero_stride_any(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    } else if (P[0] == 1.0) {
      exo_drotm_flag_one_stride_any(
          nullptr, N, exo_win_1f64{.data = X, .strides = {incX}},
          exo_win_1f64{.data = Y, .strides = {incY}}, H);
    }
  }
}
