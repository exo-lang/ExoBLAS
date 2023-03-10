#include <vector>
#include <tuple>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_dscal.h"

void test_dscal(int N, double alpha, int incX) {
    printf("Running dscal test: N = %d, alpha = %f, incX = %d\n", N, alpha, incX);
    auto X = AlignedBuffer<double>(N, incX);
    auto X_expected = X;

    exo_dscal(N, alpha, X.data(), incX);
    cblas_dscal(N, alpha, X_expected.data(), incX);

    for (int i = 0; i < X.size(); ++i) {
        if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, X_expected[i], X[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<double> alphas {0, 1, 2, -3, 3.14};
    std::vector<std::tuple<double, int> > params {{1.2, 2}, {2.5, 3}, {0, 5},
                                                 {1, 4}, {1.3, 10}, {4.5, -2}};

    for (auto n : N) {
        for (auto alpha : alphas) {
            test_dscal(n, alpha, 1);
        }
    }
    
    for (auto n : N) {
        for (auto i : params) {
            test_dscal(n, std::get<0>(i), std::get<1>(i));
        }
    }
}
