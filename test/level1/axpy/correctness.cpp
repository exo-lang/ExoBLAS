#include <vector>

#include <cblas.h>

#include "generate_buffer.h"

#include "exo_axpy.h"

void test_saxpy(int n, float alpha, int incx, int incy) {
    printf("Running saxpy test: n = %d, alpha = %f, incx = %d, incy = %d\n", n, alpha, incx, incy);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto xcopy = x;
    auto ycopy = y;

    exo_saxpy(nullptr, n, &alpha, 
    exo_win_1f32c{.data = x.data(), .strides = {incx}},
    exo_win_1f32{.data = y.data(), .strides = {incy}});
    cblas_saxpy(n, alpha, xcopy.data(), incx, ycopy.data(), incy);

    for (int i = 0; i < n; ++i) {
        if (y[i] != ycopy[i]) {
            printf("Failed ! i = %d, expected %f, got %f\n", i, ycopy[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    test_saxpy(1, 2.5, 1, 1);
    test_saxpy(2, 3.12, 1, 1);
    test_saxpy(8, 1.993, 1, 1);
    test_saxpy(100, 0, 1, 1);
    test_saxpy(64 * 64 * 64, 10.4, 1, 1);
}