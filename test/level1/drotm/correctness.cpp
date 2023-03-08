#include <vector>
#include <math.h>
#include <algorithm>
#include <tuple>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_drotm.h"

void test_drotm(int N, int incX, int incY, double HFlag) {
    printf("Running drotm test: N = %d, incX = %d, incY = %d, HFlag = %f\n", N, incX, incY, HFlag);

    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);
    auto H = generate1d_dbuffer(5, 1);
    H[0] = HFlag;
    auto X_expected = X;
    auto Y_expected = Y;
    auto H_expected = H;

    exo_drotm(N, X.data(), incX, Y.data(), incY, H.data());
    cblas_drotm(N, X_expected.data(), incX, Y_expected.data(), incY, H_expected.data());

    for (int i = 0; i < X.size(); ++i) {
        if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected X[%d] = %f, got %f\n", i, X_expected[i], X[i]);
            exit(1);
        }
    }

    for (int i = 0; i < Y.size(); ++i) {
        if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
            printf("Failed ! expected Y[%d] = %f, got %f\n", i, Y_expected[i], Y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::tuple<int, int> > params = {{2, 2}, 
                                                {3, 3},
                                                {2, 3}, 
                                                {4, 5}, 
                                                {10, -1}, 
                                                {-2, -4}};
    double HFlag[4] = {-2.0, -1.0, 0.0, 1.0};
    
    for (int i = 0; i < 4; ++i) {
        for (auto n : N) {
            test_drotm(n, 1, 1, HFlag[i]);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (auto n : N) {
            for (auto p : params) {
                test_drotm(n, std::get<0>(p), std::get<1>(p), HFlag[i]);
            }
        }
    }
}
