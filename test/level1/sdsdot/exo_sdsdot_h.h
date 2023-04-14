#pragma once

#include "exo_sdsdot.h"

float exo_sdsdot(const int N, const float alpha, const float *X, const int incX,
                 const float *Y, const int incY) {
  if (incX == 1 && incY == 1) {
    float result;
    exo_sdsdot_stride_1(nullptr, N, &alpha,
                        exo_win_1f32c{.data = X, .strides = {incX}},
                        exo_win_1f32c{.data = Y, .strides = {incY}}, &result);
    return result;
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    if (incY < 0) {
      Y = Y + (1 - N) * incY;
    }
    float result;
    exo_sdsdot_stride_any(nullptr, N, &alpha,
                          exo_win_1f32c{.data = X, .strides = {incX}},
                          exo_win_1f32c{.data = Y, .strides = {incY}}, &result);
    return result;
  }
}
