#include <vector>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_sdot.h"

void test_sdot(int N, int incX, int incY) {
    printf("Running sdot test: N = %d, incX = %d, incY = %d\n", N, incX, incY);
    auto X = AlignedBuffer<float>(N, incX);
    auto Y = AlignedBuffer<float>(N, incY);
    auto X_expected = X;
    auto Y_expected = Y;

    auto result = exo_sdot(N, X.data(), incX, Y.data(), incY);
    auto expected = cblas_sdot(N, X_expected.data(), incX, Y_expected.data(), incY);

    auto epsilon = 1.f / 100.f;

    if (!check_relative_error_okay(result, expected, epsilon)) {
        printf("Failed! Expected %f, got %f\n", expected, result);
        exit(1);
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::pair<int, int> > inc {{2, 2}, {3, 3}, {2, 3}, {4, 5}, {10, -1}, {-2, -4}};

    for (auto n : N) {
        test_sdot(n, 1, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_sdot(n, i.first, i.second);
        }
    }
}
