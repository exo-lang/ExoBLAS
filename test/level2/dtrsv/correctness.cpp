#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dtrsv.h"
#include "generate_buffer.h"

void test_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                const int N, const int lda, const int incX) {
  printf(
      "Running trsv test: N = %d, incX = %d, lda = %d, isUpper = %d, isTransA "
      "= %d, isDiag = %d\n",
      N, incX, lda, Uplo == CBLAS_UPLO::CblasUpper,
      TransA == CBLAS_TRANSPOSE::CblasTrans, Diag == CBLAS_DIAG::CblasUnit);

  auto X = AlignedBuffer<double>(N, incX);
  auto A = AlignedBuffer<double>(N * lda, 1);

  // TODO: Figure out how to pass correct inputs
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) A[i * lda + j] = 2.0;
  }
  for (int i = 0; i < X.size(); ++i) {
    X[i] = i;
  }

  auto X_expected = X;
  auto A_expected = A;

  exo_dtrsv(order, Uplo, TransA, Diag, N, A.data(), lda, X.data(), incX);
  cblas_dtrsv(order, Uplo, TransA, Diag, N, A_expected.data(), lda,
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
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<CBLAS_TRANSPOSE> transA_vals{CBLAS_TRANSPOSE::CblasNoTrans,
                                           CBLAS_TRANSPOSE::CblasTrans};
  std::vector<CBLAS_DIAG> Diag_vals{CBLAS_DIAG::CblasUnit,
                                    CBLAS_DIAG::CblasNonUnit};
  std::vector<int> lda_diffs{0, 3};
  std::vector<int> incX_vals{1, -2};

  for (auto n : N) {
    for (auto Uplo : Uplo_vals) {
      for (auto transA : transA_vals) {
        for (auto Diag : Diag_vals) {
          for (auto lda_diff : lda_diffs) {
            for (auto incX : incX_vals) {
              test_dtrsv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n,
                         n + lda_diff, incX);
            }
          }
        }
      }
    }
  }
}
