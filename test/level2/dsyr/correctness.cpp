#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dsyr.h"
#include "generate_buffer.h"

void test_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const double alpha, const int incX, const int lda) {
  printf(
      "Running dsyr test: N = %d, alpha = %f, incX = %d, lda = %d, isUpper = "
      "%d\n",
      N, alpha, incX, lda, Uplo == CBLAS_UPLO::CblasUpper);

  auto X = AlignedBuffer<double>(N, incX);
  auto A = AlignedBuffer<double>(N * lda, 1);
  auto X_expected = X;
  auto A_expected = A;

  exo_dsyr(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
  cblas_dsyr(order, Uplo, N, alpha, X_expected.data(), incX, A_expected.data(),
             lda);

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
  std::vector<CBLAS_UPLO> Uplo_values{CBLAS_UPLO::CblasUpper,
                                      CBLAS_UPLO::CblasLower};
  std::vector<int> N{3, 10, 9, 200, 2048, 3000};
  std::vector<double> alpha_vals = {1.2, -2.2, 0.0, 1.0};
  std::vector<int> incX_vals{1, 2, -3};
  std::vector<int> lda_diffs{0, 1, 5};

  for (auto Uplo : Uplo_values) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto incX : incX_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            test_dsyr(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, incX, lda);
          }
        }
      }
    }
  }
}
