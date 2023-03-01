#include <vector>

#include <cblas.h>

#include "generate_buffer.h"

#include "exo_copy.h"

void test_scopy(int n, int incx, int incy) {
    printf("Running scopy test: n = %d, incx = %d, incy = %d\n", n, incx, incy);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto xcopy = x;
    auto ycopy = y;

    float result;
    exo_scopy(nullptr, n, 
    exo_win_1f32c{.data = x.data(), .strides = {incx}},
    exo_win_1f32{.data = y.data(), .strides = {incy}});
    cblas_scopy(n, xcopy.data(), incx, ycopy.data(), incy);

    for (int i = 0; i < n; ++i) {
        if (y[i] != ycopy[i]) {
            printf("Failed ! i = %d, expected %f, got %f\n", i, ycopy[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    test_scopy(1, 1, 1);
    test_scopy(2, 1, 1);
    test_scopy(8, 1, 1);
    test_scopy(100, 1, 1);
    test_scopy(64 * 64 * 64, 1, 1);
    test_scopy(100000000, 1, 1);
}