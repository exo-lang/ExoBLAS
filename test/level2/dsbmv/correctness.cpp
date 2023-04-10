#include <vector>
#include <tuple>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_dsbmv.h"

void test_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const double alpha,
                 const int lda, const int incX,
                 const double beta, const int incY) {
    printf("Running dsbmv test: N = %d, K = %d, alpha = %f, lda = %d, incX = %d, beta = %f, incY = %d, isUpper = %d\n",
                                N, K, alpha,
                                lda, incX,
                                beta, incY,
                                Uplo == CBLAS_UPLO::CblasUpper);

    auto A = AlignedBuffer<double>(N * lda, 1);
    auto X = AlignedBuffer<double>(N, incX);
    auto Y = AlignedBuffer<double>(N, incY);

    auto A_expected = A;
    auto X_expected = X;
    auto Y_expected = Y;

    exo_dsbmv(order, Uplo, N, K, alpha, A.data(), lda, X.data(), incX, beta, Y.data(), incY);
    cblas_dsbmv(order, Uplo, N, K, alpha, A_expected.data(), lda, X_expected.data(), incX, beta, Y_expected.data(), incY);

    for (int i = 0; i < Y.size(); ++i) {
        if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 100.f)) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, Y_expected[i], Y[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<CBLAS_UPLO> Uplo_vals {CBLAS_UPLO::CblasUpper, CBLAS_UPLO::CblasLower};
    std::vector<int> N {4, 41, 80, 101, 1024};
    std::vector<int> K {0, 1, 2, 50, 100};
    std::vector<double> alpha_vals {0, 1, 2.4};
    std::vector<int> lda_diffs {0, 3, 5};
    std::vector<int> incX_vals {1, 3, -2};
    std::vector<double> beta_vals {0, 1, 2.4};
    std::vector<int> incY_vals {1, 3, -2};

    for (auto Uplo : Uplo_vals) {
        for (auto n : N) {
            for (auto alpha : alpha_vals) {
                for (auto lda_diff : lda_diffs) {
                    for (auto incX : incX_vals) {
                        for (auto beta : beta_vals) {
                            for (auto incY : incY_vals) {
                                for (auto k : K) {
                                    if (k <= n - 1) {
                                        int lda = k + 1 + lda_diff;
                                        test_dsbmv(CBLAS_ORDER::CblasRowMajor,
                                        Uplo, n, k, alpha, lda, incX, beta, incY);
                                    }
                                }
                                int lda = n + lda_diff;
                                test_dsbmv(CBLAS_ORDER::CblasRowMajor,
                                        Uplo, n, n - 1, alpha, lda, incX, beta, incY);
                            }
                        }
                    }
                }
            }
        }
    }
}