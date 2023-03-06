#include <vector>

#include <cblas.h>

#include "generate_buffer.h"

#include "exo_dswap.h"

void test_dswap(int N, int incX, int incY) {
    printf("Running dswap test: N = %d, incX = %d, incY = %d\n", N, incX, incY);
    auto X = generate1d_dbuffer(N, incX);
    auto Y = generate1d_dbuffer(N, incY);
    auto X_expected = X;
    auto Y_expected = Y;

    exo_dswap(N, X.data(), incX, Y.data(), incY);
    cblas_dswap(N, X_expected.data(), incX, Y_expected.data(), incY);

    for (int i = 0; i < Y.size(); ++i) {
        if (Y[i] != Y_expected[i]) {
            printf("Failed ! memory offset = %d, expected %f, got %f\n", i, Y_expected[i], Y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::pair<int, int> > inc {{2, 2}, {3, 3}, {2, 3}, {4, 5}, {10, -1}, {-2, -4}};

    for (auto n : N) {
        test_dswap(n, 1, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_dswap(n, i.first, i.second);
        }
    }
}