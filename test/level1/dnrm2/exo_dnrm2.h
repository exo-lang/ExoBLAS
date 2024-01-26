#pragma once

#include <math.h>

#include "blue_algorithm.h"
#include "exo_nrm2.h"

double exo_dnrm2(const int N, const double *X, const int incX) {
  double t_sml = blue_algorithm<double>::t_sml;
  double t_big = blue_algorithm<double>::t_big;
  double s_sml = blue_algorithm<double>::s_sml;
  double s_big = blue_algorithm<double>::s_big;

  double a_sml;
  double a_med;
  double a_big;

  if (incX == 1) {
    exo_dnrm2_stride_1(nullptr, N, exo_win_1f64c{.data = X, .strides = {incX}},
                       &t_sml, &t_big, &s_sml, &s_big, &a_sml, &a_med, &a_big);
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    exo_dnrm2_stride_any(nullptr, N,
                         exo_win_1f64c{.data = X, .strides = {incX}}, &t_sml,
                         &t_big, &s_sml, &s_big, &a_sml, &a_med, &a_big);
  }

  return blue_algorithm<double>::combine_accums(a_sml, a_med, a_big);
}
