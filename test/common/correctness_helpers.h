#pragma once

#include <algorithm>
#include <math.h>

bool check_relative_error_okay(double result, double expected, double epsilon) {
    auto error = std::fabs(result - expected);
    auto rel_error = error / std::max((double)1.0, std::fabs(expected));
    return rel_error < epsilon;
}
