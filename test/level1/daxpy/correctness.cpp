#include <vector>
#include <tuple>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_daxpy.h"

void test_daxpy(int N, double alpha, int incX, int incY) {
    printf("Running daxpy test: N = %d, alpha = %f, incX = %d, incY = %d\n", N, alpha, incX, incY);
    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);
    auto X_expected = X;
    auto Y_expected = Y;

    exo_daxpy(N, alpha, X.data(), incX, Y.data(), incY);
    cblas_daxpy(N, alpha, X_expected.data(), incX, Y_expected.data(), incY);

    for (int i = 0; i < Y.size(); ++i) {
        if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, Y_expected[i], Y[i]);
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