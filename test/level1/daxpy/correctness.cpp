#include <vector>
#include <tuple>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_daxpy_wrapper.h"

void test_daxpy(int n, double alpha, int incx, int incy) {
    printf("Running daxpy test: n = %d, alpha = %f, incx = %d, incy = %d\n", n, alpha, incx, incy);
    auto x = generate1d_dbuffer(n, incx);
    auto y = generate1d_dbuffer(n, incy);
    auto xcopy = x;
    auto ycopy = y;

    exo_daxpy(n, alpha, x.data(), incx, y.data(), incy);
    cblas_daxpy(n, alpha, xcopy.data(), incx, ycopy.data(), incy);

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(y[i], ycopy[i], 1.f / 10000.f)) {
            printf("Failed ! i = %d, expected %f, got %f\n", i, ycopy[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::tuple<double, int, int> > params {{1.2, 2, 2}, {2.5, 3, 3}, {0, 2, 3},
                                                 {1, 4, 5}, {1.3, 10, -1}, {4.5, -2, -4}};

    for (auto n : N) {
        test_daxpy(n, 1.2, 1, 1);
    }
    
    for (auto n : N) {
        for (auto i : params) {
            test_daxpy(n, std::get<0>(i), std::get<1>(i), std::get<2>(i));
        }
    }
}