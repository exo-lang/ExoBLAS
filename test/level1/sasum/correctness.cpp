#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_sasum.h"

void test_sasum(int N, int incX) {
    printf("Running sasum test: N = %d, incX = %d\n", N, incX);
    auto X = AlignedBuffer<float>(N, incX);
    auto X_expected = X;

    auto result = exo_sasum(N, X.data(), incX);
    auto expected = cblas_sasum(N, X_expected.data(), incX);

    auto epsilon = 1.f / 100.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<int> inc {2, 3, 5, 10};

    for (auto n : N) {
        test_sasum(n, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_sasum(n, i);
        }
    }
}
