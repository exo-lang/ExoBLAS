#include <vector>

#include <cblas.h>

#include "generate_buffer.h"

#include "exo_scopy.h"
 
void test_scopy(int N, int incX, int incY) {
    printf("Running scopy test: N = %d, incX = %d, incY = %d\n", N, incX, incY);
    auto X = generate1d_sbuffer(N, incX);
    auto Y = generate1d_sbuffer(N, incY);
    auto X_expected = X;
    auto Y_expected = Y;

    exo_scopy(N, X.data(), incX, Y.data(), incY);
    cblas_scopy(N, X_expected.data(), incX, Y_expected.data(), incY);

    for (int i = 0; i < Y_expected.size(); ++i) {
        if (Y_expected[i] != Y[i]) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, Y_expected[i], Y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {1, 2, 8, 100, 64 * 64 * 64, 10000000};
    std::vector<std::pair<int, int> > inc {{2, 2}, {3, 3}, {2, 3}, {4, 5}, {10, -1}, {-2, -4}};

    for (auto n : N) {
        test_scopy(n, 1, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_scopy(n, i.first, i.second);
        }
    }
}