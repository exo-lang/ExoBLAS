#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dsymv.h"
#include "generate_buffer.h"

void test_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const int lda, const int incX,
                const double beta, const int incY) {
  printf(
      "Running dsymv test: N = %d, alpha = %f, lda = %d, incX = %d, beta = %f, "
      "incY = %d, isUpper = %d\n",
      N, alpha, lda, incX, beta, incY, Uplo == CBLAS_UPLO::CblasUpper);

  auto A = AlignedBuffer<double>(N * lda, 1);
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);

  auto A_expected = A;
  auto X_expected = X;
  auto Y_expected = Y;

  exo_dsymv(order, Uplo, N, alpha, A.data(), lda, X.data(), incX, beta,
            Y.data(), incY);
  cblas_dsymv(order, Uplo, N, alpha, A_expected.data(), lda, X_expected.data(),
              incX, beta, Y_expected.data(), incY);

  for (int i = 0; i < Y.size(); ++i) {
    if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 100.f)) {
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             Y_expected[i], Y[i]);
      exit(1);
    }
  }

  printf("Passed!\n");
}

int main() {
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<int> N{2, 100, 321};
  std::vector<double> alpha_vals{1, 2.4};
  std::vector<int> lda_diffs{0, 3};
  std::vector<int> incX_vals{1, 3};
  std::vector<double> beta_vals{0, 2.4};
  std::vector<int> incY_vals{1, -2};

  for (auto Uplo : Uplo_vals) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto lda_diff : lda_diffs) {
          int lda = n + lda_diff;
          for (auto incX : incX_vals) {
            for (auto beta : beta_vals) {
              for (auto incY : incY_vals) {
                test_dsymv(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, lda,
                           incX, beta, incY);
              }
            }
          }
        }
      }
    }
  }
}
