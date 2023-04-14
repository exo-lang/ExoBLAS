#pragma once

#include <math.h>

#include "exo_nrm2.h"

double exo_dnrm2(const int N, const double *X, const int incX) {
  if (incX == 1) {
    double result;
    exo_dnrm2_stride_1(nullptr, N, exo_win_1f64c{.data = X, .strides = {incX}},
                       &result);
    return sqrt(result);
  } else {
    if (incX < 0) {
      X = X + (1 - N) * incX;
    }
    double result;
    exo_dnrm2_stride_any(nullptr, N,
                         exo_win_1f64c{.data = X, .strides = {incX}}, &result);
    return sqrt(result);
  }
}
