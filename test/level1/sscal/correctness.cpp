#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_sscal.h"

void test_sscal(int N, float alpha, int incX) {
    printf("Running sscal test: N = %d, alpha = %f, incX = %d\n", N, alpha, incX);
    auto X = generate1d_sbuffer(N, incX);
    auto X_expected = X;

    exo_sscal(N, alpha, X.data(), incX);
    cblas_sscal(N, alpha, X_expected.data(), incX);

    for (int i = 0; i < X.size(); ++i) {
        if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
            printf("Failed ! memory offset = %d, expected %f, got %f\n", i, X_expected[i], X[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<float> alphas {0, 1, 2, -3, 3.14};
    std::vector<std::tuple<float, int> > params {{1.2, 2}, {2.5, 3}, {0, -1},
                                                 {1, 4}, {1.3, 10}, {4.5, -2}};

    for (auto n : N) {
        for (auto alpha : alphas) {
            test_sscal(n, alpha, 1);
        }
    }
    
    for (auto n : N) {
        for (auto i : params) {
            test_sscal(n, std::get<0>(i), std::get<1>(i));
        }
    }
}
