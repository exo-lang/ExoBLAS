#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_scal.h"

void test_sscal(int n, int incx, float alpha) {
    printf("Running sscal test: n = %d, incx = %d, alpha = %f\n", n, incx, alpha);
    auto x = generate1d_sbuffer(n, incx);
    auto xcopy = x;

    exo_sscal(nullptr, n, &alpha, exo_win_1f32{.data = x.data(), .strides = {incx}});
    cblas_sscal(n, alpha, xcopy.data(), incx);

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(x[i], xcopy[i], 1.f / 10000.f)) {
            printf("Failed ! i = %d, expected %f, got %f\n", i, xcopy[i], x[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    test_sscal(1, 1, 2.5);
    test_sscal(2, 1, 3.12);
    test_sscal(8, 1, 1.993);
    test_sscal(100, 1, 0);
    test_sscal(64 * 64 * 64, 1, 10.4);
}