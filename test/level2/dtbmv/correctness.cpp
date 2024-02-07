#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dtbmv.h"
#include "generate_buffer.h"

void test_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                const int N, const int k, const int lda, const int incX) {
  printf(
      "Running dtbmv test: N = %d, k = %d, incX = %d, lda = %d, isUpper = %d, "
      "isTransA = %d, isDiag = %d\n",
      N, k, incX, lda, Uplo == CBLAS_UPLO::CblasUpper,
      TransA == CBLAS_TRANSPOSE::CblasTrans, Diag == CBLAS_DIAG::CblasUnit);
  auto X = AlignedBuffer<double>(N, incX);
  auto A = AlignedBuffer<double>(N * lda, 1);

  auto X_expected = X;
  auto A_expected = A;

  exo_dtbmv(order, Uplo, TransA, Diag, N, k, A.data(), lda, X.data(), incX);
  cblas_dtbmv(order, Uplo, TransA, Diag, N, k, A_expected.data(), lda,
              X_expected.data(), incX);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 100.f)) {
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             X_expected[i], X[i]);
      exit(1);
    }
  }

  printf("Passed!\n");
}

int main() {
  std::vector<int> N{2, 100, 321};
  std::vector<int> K{0, 1, 6};
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<CBLAS_TRANSPOSE> transA_vals{CBLAS_TRANSPOSE::CblasNoTrans,
                                           CBLAS_TRANSPOSE::CblasTrans};
  std::vector<CBLAS_DIAG> Diag_vals{CBLAS_DIAG::CblasNonUnit,
                                    CBLAS_DIAG::CblasUnit};
  std::vector<int> lda_diffs{0, 3};
  std::vector<int> incX_vals{1, 3};

  for (auto n : N) {
    for (auto Uplo : Uplo_vals) {
      for (auto transA : transA_vals) {
        for (auto Diag : Diag_vals) {
          for (auto lda_diff : lda_diffs) {
            for (auto incX : incX_vals) {
              for (auto k : K) {
                if (k <= n - 1) {
                  test_dtbmv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n,
                             k, k + 1 + lda_diff, incX);
                }
              }
              test_dtbmv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n,
                         n - 1, n + lda_diff, incX);
            }
          }
        }
      }
    }
  }
}
