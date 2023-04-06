#include <vector>
#include <tuple>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_stbsv.h"

void test_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                const int N, int K,
                const int lda, const int incX) {
    printf("Running stbsv test: N = %d, K = %d, incX = %d, lda = %d, isUpper = %d, isTransA = %d, isDiag = %d\n", N, K, incX, lda, 
        Uplo == CBLAS_UPLO::CblasUpper,
        TransA == CBLAS_TRANSPOSE::CblasTrans,
        Diag == CBLAS_DIAG::CblasUnit);

    auto X = AlignedBuffer<float>(N, incX);
    auto A = AlignedBuffer<float>(N * lda, 1);

    // Make sure the dot product is on the order of
    // the values in the rhs of the system
    for (int i = 0; i < A.size(); ++i) {
        A[i] /= 1000;
    }
    // Make sure that the pivots are > 1 so that
    // the values of the variables don't keep increasing
    for (int i = 0; i < N; ++i) {
        if (Uplo == CBLAS_UPLO::CblasUpper) {
            A[i * lda] = 2;
        } else {
            A[i * lda + K] = 2;
        }
    }

    auto X_expected = X;
    auto A_expected = A;

    exo_stbsv(order, Uplo, TransA, Diag, N, K, A.data(), lda, X.data(), incX);
    cblas_stbsv(order, Uplo, TransA, Diag, N, K, A_expected.data(), lda, X_expected.data(), incX);

    for (int i = 0; i < X.size(); ++i) {
        if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 100.f)) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, X_expected[i], X[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> N {4, 20, 50, 100, 1024};
    std::vector<int> K {0, 1, 2, 10, 60};
    std::vector<CBLAS_UPLO> Uplo_vals {CBLAS_UPLO::CblasUpper, CBLAS_UPLO::CblasLower};
    std::vector<CBLAS_TRANSPOSE> transA_vals {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans};
    std::vector<CBLAS_DIAG> Diag_vals {CBLAS_DIAG::CblasUnit, CBLAS_DIAG::CblasNonUnit};
    std::vector<int> lda_diffs {0, 3, 5};
    std::vector<int> incX_vals {1, 3, -2};

    for (auto n : N) {            
        for (auto Uplo : Uplo_vals) {
            for (auto transA : transA_vals) {
                for (auto Diag: Diag_vals) {
                    for (auto lda_diff : lda_diffs) {
                        for (auto incX : incX_vals) {
                            for (auto k : K) {
                                if (k <= n - 1) {
                                    test_stbsv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n, k, k + 1 + lda_diff, incX);
                                }
                                test_stbsv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n, n - 1, n + lda_diff, incX);
                            }
                        }
                    }
                }
            }
        }
    }
}