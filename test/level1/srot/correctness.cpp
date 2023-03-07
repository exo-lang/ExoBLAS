#include <vector>
#include <math.h>
#include <algorithm>
#include <tuple>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_srot.h"

void test_srot(int N, int incX, int incY, float c, float s) {
    printf("Running srot test: N = %d, incX = %d, incY = %d, c = %f, s = %f\n", N, incX, incY, c, s);
    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);
    auto X_expected = X;
    auto Y_expected = Y;

    exo_srot(N, X.data(), incX, Y.data(), incY, c, s);
    cblas_srot(N, X_expected.data(), incX, Y_expected.data(), incY, c, s);

    for (int i = 0; i < N; ++i) {
        if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected X[%d] = %f, got %f\n", i, X_expected[i], X[i]);
            exit(1);
        }
    }

    for (int i = 0; i < N; ++i) {
        if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected Y[%d] = %f, got %f\n", i, Y_expected[i], Y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::tuple<float, float> > consts = {
                                                     {2, 3},
                                                     {1, 1},
                                                     {0, 0},
                                                     {-1, 1},
                                                    };
    std::vector<std::tuple<int, int, float, float> > params {{2, 2, 1, 2}, 
                                                          {3, 3, -1, 1},
                                                          {2, 3, 2.123, 4.21}, 
                                                          {4, 5, 0, -0.123}, 
                                                          {10, -1, 2, 0}, 
                                                          {-2, -4, 0.03, 1.2}};
    
    for (auto n : N) {
        for (auto cp : consts) {
            test_srot(n, 1, 1, std::get<0>(cp), std::get<1>(cp));
        }
    }

    for (auto n : N) {
        for (auto param : params) {
            test_srot(n, std::get<0>(param), std::get<1>(param),
                         std::get<2>(param), std::get<3>(param));
        }
    }
}