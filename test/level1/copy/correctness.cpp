#include <vector>

#include <cblas.h>

#include "generate_buffer.h"

#include "exo_copy_wrapper.h"
 
void test_scopy(int n, int incx, int incy) {
    printf("Running scopy test: n = %d, incx = %d, incy = %d\n", n, incx, incy);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto x_expected = x;
    auto y_expected = y;

    exo_scopy(n, x.data(), incx, y.data(), incy);
    cblas_scopy(n, x_expected.data(), incx, y_expected.data(), incy);

    for (int i = 0; i < y_expected.size(); ++i) {
        if (y_expected[i] != y[i]) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, y_expected[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

void test_dcopy(int n, int incx, int incy) {
    printf("Running dcopy test: n = %d, incx = %d, incy = %d\n", n, incx, incy);
    auto x = generate1d_dbuffer(n, incx);
    auto y = generate1d_dbuffer(n, incy);
    auto x_expected = x;
    auto y_expected = y;

    exo_dcopy(n, x.data(), incx, y.data(), incy);
    cblas_dcopy(n, x_expected.data(), incx, y_expected.data(), incy);

    for (int i = 0; i < y_expected.size(); ++i) {
        if (y_expected[i] != y[i]) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, y_expected[i], y[i]);
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

    for (auto n : N) {
        test_dcopy(n, 1, 1);
    }
    
    for (auto n : N) {
        for (auto i : inc) {
            test_dcopy(n, i.first, i.second);
        }
    }
}