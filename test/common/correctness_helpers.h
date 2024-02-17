#pragma once

#include <algorithm>
#include <cmath>

double constexpr default_error = 1.0 / 100.0;

bool check_relative_error_okay(double result, double expected,
                               double epsilon = default_error) {
  auto error = std::fabs(result - expected);
  auto rel_error = error / std::max((double)1.0, std::fabs(expected));
  return rel_error < epsilon;
}
