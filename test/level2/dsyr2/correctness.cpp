#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dsyr2.h"
#include "generate_buffer.h"

void test_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, int N,
                double alpha, int incX, int incY, int lda) {
  printf(
      "Running dsyr2 test: N = %d, alpha = %f, incX = %d, incY = %d, lda = %d, "
      "isUpper = %d\n",
      N, alpha, incX, incY, lda, Uplo == CBLAS_UPLO::CblasUpper);
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto A = AlignedBuffer<double>(N * lda, 1);
  auto X_expected = X;
  auto Y_expected = Y;
  auto A_expected = A;

  exo_dsyr2(CBLAS_ORDER::CblasRowMajor, Uplo, N, alpha, X.data(), incX,
            Y.data(), incY, A.data(), lda);
  cblas_dsyr2(CBLAS_ORDER::CblasRowMajor, Uplo, N, alpha, X_expected.data(),
              incX, Y_expected.data(), incY, A_expected.data(), lda);

  for (int i = 0; i < A.size(); ++i) {
    if (!check_relative_error_okay(A[i], A_expected[i], 1.f / 10000.f)) {
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             A_expected[i], A[i]);
      exit(1);
    }
  }

  printf("Passed!\n");
}

int main() {
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<int> N{3, 10, 9, 200, 2048};
  std::vector<double> alpha_vals{1.2, 0.0, 1.0};
  std::vector<int> incX_vals{1, 2, -3};
  std::vector<int> incY_vals{1, 2, -3};
  std::vector<int> lda_diffs{0, 1, 5};

  for (auto Uplo : Uplo_vals) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto incX : incX_vals) {
          for (auto incY : incY_vals) {
            for (auto lda_diff : lda_diffs) {
              int lda = n + lda_diff;
              test_dsyr2(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, incX, incY,
                         lda);
            }
          }
        }
      }
    }
  }
}
