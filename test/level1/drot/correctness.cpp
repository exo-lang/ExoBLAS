#include <vector>
#include <math.h>
#include <algorithm>
#include <tuple>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_drot.h"

void test_drot(int N, int incX, int incY, double c, double s) {
    printf("Running drot test: N = %d, incX = %d, incY = %d, c = %f, s = %f\n", N, incX, incY, c, s);
    auto x = generate1d_dbuffer(N, incX);
    auto y = generate1d_dbuffer(N, incY);
    auto x_expected = x;
    auto y_expected = y;

    exo_drot(N, x.data(), incX, y.data(), incY, c, s);
    cblas_drot(N, x_expected.data(), incX, y_expected.data(), incY, c, s);

    for (int i = 0; i < N; ++i) {
        if (!check_relative_error_okay(x[i], x_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected x[%d] = %f, got %f\n", i, x_expected[i], x[i]);
            exit(1);
        }
    }

    for (int i = 0; i < N; ++i) {
        if (!check_relative_error_okay(y[i], y_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected y[%d] = %f, got %f\n", i, y_expected[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::tuple<double, double> > consts = {
                                                     {2, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {-1, 1},
                                                    };
    std::vector<std::tuple<int, int, double, double> > params {{2, 2, 1, 2}, 
                                                          {3, 3, -1, 1},
                                                          {2, 3, 2.123, 4.21}, 
                                                          {4, 5, 0, -0.123}, 
                                                          {10, -1, 2, 0}, 
                                                          {-2, -4, 0.03, 1.2}};
    
    for (auto n : N) {
        for (auto cp : consts) {
            test_drot(n, 1, 1, std::get<0>(cp), std::get<1>(cp));
        }
    }

    for (auto n : N) {
        for (auto param : params) {
            test_drot(n, std::get<0>(param), std::get<1>(param),
                         std::get<2>(param), std::get<3>(param));
        }
    }
}