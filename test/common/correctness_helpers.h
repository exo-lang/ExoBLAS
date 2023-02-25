#pragma once

#include <algorithm>
#include <math.h>

bool check_relative_error_okay(float result, float expected, float epsilon) {
    auto error = fabsf(result - expected);
    auto rel_error = error / std::max(1.f, fabsf(expected));
    return rel_error < epsilon;
}
