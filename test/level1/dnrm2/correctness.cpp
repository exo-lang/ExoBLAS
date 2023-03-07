#include <vector>
#include <math.h>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_dnrm2.h"

void test_dnrm2(int N, int incX) {
    printf("Running dnrm2 test: N = %d, incX = %d\n", N, incX);
    auto X = generate1d_dbuffer(N, incX);
    auto X_expected = X;

    auto result = exo_dnrm2(N, X.data(), incX);
    auto expected = cblas_dnrm2(N, X_expected.data(), incX);

    auto epsilon = 1.f / 1000.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }
    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 1000000};
    std::vector<int> inc {2, 3, 5, -1, -2, -10};

    for (auto n : N) {
        test_dnrm2(n, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_dnrm2(n, i);
        }
    }
}