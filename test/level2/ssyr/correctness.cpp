#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_ssyr.h"
#include "generate_buffer.h"

void test_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const float alpha, const int incX, const int lda) {
  printf(
      "Running ssyr test: N = %d, alpha = %f, incX = %d, lda = %d, isUpper = "
      "%d\n",
      N, alpha, incX, lda, Uplo == CBLAS_UPLO::CblasUpper);

  auto X = AlignedBuffer<float>(N, incX);
  auto A = AlignedBuffer<float>(N * lda, 1);
  auto X_expected = X;
  auto A_expected = A;

  exo_ssyr(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
  cblas_ssyr(order, Uplo, N, alpha, X_expected.data(), incX, A_expected.data(),
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
  std::vector<int> N{2, 100, 321};
  std::vector<float> alpha_vals = {1.2};
  std::vector<int> incX_vals{1, 2};
  std::vector<int> lda_diffs{0, 5};

  for (auto Uplo : Uplo_values) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto incX : incX_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            test_ssyr(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, incX, lda);
          }
        }
      }
    }
  }
}
