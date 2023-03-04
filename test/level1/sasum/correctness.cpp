#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_asum.h"

void test_sasum(int n, int incx) {
    printf("Running sasum test: n = %d, incx = %d\n", n, incx);
    auto x = generate1d_sbuffer(n, incx);
    auto xcopy = x;

    float result;
    exo_sasum(nullptr, n, 
    exo_win_1f32c{.data = x.data(), .strides = {incx}},
    &result);
    auto expected = cblas_sasum(n, xcopy.data(), incx);

    auto epsilon = 1.f / 1000.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }
    printf("Expected %f, got %f\n", expected, result);
    printf("Passed!\n");
}

int main () {
    test_sasum(1, 1);
    test_sasum(2, 1);
    test_sasum(8, 1);
    test_sasum(100, 1);
    test_sasum(64 * 64 * 64, 1);
    test_sasum(10000000, 1);
}