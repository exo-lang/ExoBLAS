#include <vector>
#include <math.h>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_rotm.h"

void test_srotm(int n, int incx, int incy, float HFlag) {
    printf("Running srotm test: n = %d, incx = %d, incy = %d, HFlag = %f\n", n, incx, incy, HFlag);
    auto x = generate1d_sbuffer(n, incx);
    auto y = generate1d_sbuffer(n, incy);
    auto H = generate1d_sbuffer(4, 1);
    auto xcopy = x;
    auto ycopy = y;
    auto Hcopy = generate1d_sbuffer(n, 5);
    Hcopy[0] = HFlag;
    // CBLAS has H in colum-major order!!!
    Hcopy[1] = H[0];
    Hcopy[2] = H[2];
    Hcopy[3] = H[1];
    Hcopy[4] = H[3];

    exo_srotm(nullptr, n, 
    exo_win_1f32{.data = x.data(), .strides = {incx}},
    exo_win_1f32{.data = y.data(), .strides = {incy}}, (int_fast32_t)HFlag, H.data());
    cblas_srotm(n, xcopy.data(), incx, ycopy.data(), incy, Hcopy.data());

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(x[i], xcopy[i], 1.f / 10000.f)) {
            printf("Failed ! expected x[%d] = %f, got %f\n", i, xcopy[i], x[i]);
            exit(1);
        }
    }

    for (int i = 0; i < n; ++i) {
        if (!check_relative_error_okay(y[i], ycopy[i], 1.f / 10000.f)) {
            printf("Failed ! expected y[%d] = %f, got %f\n", i, ycopy[i], y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    float HFlag[4] = {-2.0, -1.0, 0.0, 1.0};
    for (int i = 0; i < 4; ++i) {
        test_srotm(1, 1, 1, HFlag[i]);
        test_srotm(2, 1, 1, HFlag[i]);
        test_srotm(8, 1, 1, HFlag[i]);
        test_srotm(100, 1, 1, HFlag[i]);
        test_srotm(64 * 64 * 64, 1, 1, HFlag[i]);
        test_srotm(100000000, 1, 1, HFlag[i]);
    }
}