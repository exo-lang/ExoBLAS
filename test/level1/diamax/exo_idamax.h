#pragma once

#include <stdint.h>

#include "exo_iamax.h"

int32_t exo_idamax(const int N, const double *X, const int incX) {
    if (incX == 1) {
        int32_t index;
        exo_idamax_stride_1(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            &index);
        return index;
    } else {
        if (incX <= 0) {
            return 0;
        }
        int32_t index;
        exo_idamax_stride_any(nullptr, N, 
            exo_win_1f64c{.data = X, .strides = {incX}},
            &index);
        return index;
    }
}
