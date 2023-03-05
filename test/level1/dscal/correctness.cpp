#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_dscal.h"

void test_dscal(int N, double alpha, int incX) {
    printf("Running dscal test: N = %d, alpha = %f, incX = %d\n", N, alpha, incX);
    auto x = generate1d_dbuffer(N, incX);
    auto x_expected = x;

    exo_dscal(N, alpha, x.data(), incX);
    cblas_dscal(N, alpha, x_expected.data(), incX);

    for (int i = 0; i < x.size(); ++i) {
        if (!check_relative_error_okay(x[i], x_expected[i], 1.f / 10000.f)) {
            printf("Failed ! memory offset = %d, expected %f, got %f\n", i, x_expected[i], x[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<double> alphas {0, 1, 2, -3, 3.14};
    std::vector<std::tuple<double, int> > params {{1.2, 2}, {2.5, 3}, {0, -1},
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
