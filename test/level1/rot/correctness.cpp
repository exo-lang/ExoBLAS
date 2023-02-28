#include <vector>
#include <math.h>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_rot.h"

void test_srot(int n, int incx, int incy, float c, float s) {
    printf("Running srot test: n = %d, incx = %d, incy = %d, c = %f, s = %f\n", n, incx, incy, c, s);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto xcopy = x;
    auto ycopy = y;

    float result;
    exo_srot(nullptr, n, 
    exo_win_1f32{.data = x.data(), .strides = {incx}},
    exo_win_1f32{.data = y.data(), .strides = {incy}}, &c, &s);
    cblas_srot(n, xcopy.data(), incx, ycopy.data(), incy, c, s);

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(x[i], xcopy[i], 1.f / 10000.f)) {
            printf("Failed ! expected x[%d] = %f, got %f\n", i, xcopy[i], x[i]);
            exit(1);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(y[i], ycopy[i], 1.f / 10000.f)) {
            printf("Failed ! expected y[%d] = %f, got %f\n", i, ycopy[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    test_srot(1, 1, 1, 1.2, -1.3);
    test_srot(2, 1, 1, 2.44, 10.1);
    test_srot(8, 1, 1, 1.001, 3.123);
    test_srot(100, 1, 1, 312, -1233);
    test_srot(64 * 64 * 64, 1, 1, 123.2, 98.015);
    test_srot(100000000, 1, 1, 65.64, 32.12);
}