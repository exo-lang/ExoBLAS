#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_sdsdot.h"

void test_sdsdot(int n, int incx, int incy) {
    printf("Running sdsdot test: n = %d, incx = %d, incy = %d\n", n, incx, incy);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto xcopy = x;
    auto ycopy = y;
    float sb = 1;

    float result;
    exo_sdsdot(nullptr, n, &sb,
    exo_win_1f32c{.data = x.data(), .strides = {incx}},
    exo_win_1f32c{.data = y.data(), .strides = {incy}}, &result);
    auto expected = cblas_sdsdot(n, sb, xcopy.data(), incx, ycopy.data(), incy);

    auto epsilon = 1.f / 1000.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }

    printf("Passed!\n");
}

int main () {
    test_sdsdot(1, 1, 1);
    test_sdsdot(2, 1, 1);
    test_sdsdot(8, 1, 1);
    test_sdsdot(100, 1, 1);
    test_sdsdot(64 * 64 * 64, 1, 1);
    test_sdsdot(100000000, 1, 1);
}