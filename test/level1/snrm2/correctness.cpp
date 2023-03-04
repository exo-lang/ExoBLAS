#include <vector>
#include <math.h>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_nrm2.h"

void test_snrm2(int n, int incx) {
    printf("Running snrm2 test: n = %d, incx = %d\n", n, incx);
    auto x = generate1d_sbuffer(n, incx);
    auto xcopy = x;

    float result;
    exo_snrm2(nullptr, n, 
    exo_win_1f32c{.data = x.data(), .strides = {incx}},
    &result);
    result = sqrtf32(result);
    auto expected = cblas_snrm2(n, xcopy.data(), incx);

    auto epsilon = 1.f / 1000.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }
    printf("Expected %f, got %f\n", expected, result);
    printf("Passed!\n");
}

int main () {
    test_snrm2(1, 1);
    test_snrm2(2, 1);
    test_snrm2(8, 1);
    test_snrm2(100, 1);
    test_snrm2(64 * 64 * 64, 1);
    test_snrm2(10000000, 1);
}