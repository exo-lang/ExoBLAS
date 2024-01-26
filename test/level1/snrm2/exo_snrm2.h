#pragma once

#include <math.h>

#include "blue_algorithm.h"
#include "exo_nrm2.h"

float exo_snrm2(const int N, const float *X, const int incX) {
  float t_sml = blue_algorithm<float>::t_sml;
  float t_big = blue_algorithm<float>::t_big;
  float s_sml = blue_algorithm<float>::s_sml;
  float s_big = blue_algorithm<float>::s_big;

  float a_sml;
  float a_med;
  float a_big;

  if (incX == 1) {
    exo_snrm2_stride_1(nullptr, N, exo_win_1f32c{.data = X, .strides = {incX}},
                       &t_sml, &t_big, &s_sml, &s_big, &a_sml, &a_med, &a_big);
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    exo_snrm2_stride_any(nullptr, N,
                         exo_win_1f32c{.data = X, .strides = {incX}}, &t_sml,
                         &t_big, &s_sml, &s_big, &a_sml, &a_med, &a_big);
  }

  return blue_algorithm<float>::combine_accums(a_sml, a_med, a_big);
}
