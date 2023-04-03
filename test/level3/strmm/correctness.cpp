#include <vector>
#include <tuple>
#include <algorithm>

#include <cblas.h>

#include "generate_buffer.h"
#include "correctness_helpers.h"

#include "exo_strmm.h"

void test_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const int lda,
                 const int ldb) {
    printf("Running strmm test: M = %d, N = %d, alpha = %f, lda = %d, ldb = %d, isSideLeft = %d, isUpper = %d, isTransA = %d, isDiag = %d\n", M, N, alpha, lda, ldb, 
        Side == CBLAS_SIDE::CblasLeft,
        Uplo == CBLAS_UPLO::CblasUpper,
        TransA == CBLAS_TRANSPOSE::CblasTrans,
        Diag == CBLAS_DIAG::CblasUnit);
    int K = Side == CBLAS_SIDE::CblasLeft ? M : N;
    auto A = AlignedBuffer<float>(K * lda, 1);
    auto B = AlignedBuffer<float>(M * ldb, 1);

    auto A_expected = A;
    auto B_expected = B;

    exo_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A.data(), lda, B.data(), ldb);
    cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A_expected.data(), lda, B_expected.data(), ldb);

    for (int i = 0; i < B.size(); ++i) {
        if (!check_relative_error_okay(B[i], B_expected[i], 1.f / 100.f)) {
            printf("Failed ! mem offset = %d, expected %f, got %f\n", i, B_expected[i], B[i]);
            exit(1);
        }
    }

    printf("Passed!\n");
}

int main () {
    std::vector<int> M {4, 41, 80, 101, 1024};
    std::vector<int> N {8, 15, 304, 564};
    std::vector<CBLAS_SIDE> Side_vals {CBLAS_SIDE::CblasLeft};
    std::vector<CBLAS_UPLO> Uplo_vals {CBLAS_UPLO::CblasUpper, CBLAS_UPLO::CblasLower};
    std::vector<CBLAS_TRANSPOSE> transA_vals {CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans};
    std::vector<CBLAS_DIAG> Diag_vals {CBLAS_DIAG::CblasNonUnit, CBLAS_DIAG::CblasUnit};
    std::vector<float> alpha_vals {0, 1, 1.4};
    std::vector<int> ld_diffs {0,   5};

    for (auto m : M) {
        for (auto n : N) {
            for (auto Side : Side_vals) {
                for (auto Uplo : Uplo_vals) {
                    for (auto transA : transA_vals) {
                        for (auto Diag: Diag_vals) {
                            for (auto alpha : alpha_vals) {
                                for (auto lda_diff : ld_diffs) {
                                    for (auto ldb_diff : ld_diffs) {
                                        int K = Side == CBLAS_SIDE::CblasLeft ? m : n;
                                        int lda = K + lda_diff;
                                        int ldb = n + ldb_diff;
                                        test_strmm(CBLAS_ORDER::CblasRowMajor, Side, Uplo, transA, Diag, m, n, alpha, lda, ldb);
                                    }
                                }
                            }
                        }
                    }
                }   
            }   
        }
    }
}