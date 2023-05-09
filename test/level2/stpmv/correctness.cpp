#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_stpmv.h"
#include "generate_buffer.h"

void test_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                const int N, const int incX) {
  printf(
      "Running tpmv test: N = %d, incX = %d, isUpper = %d, isTransA "
      "= %d, isDiag = %d\n",
      N, incX, Uplo == CBLAS_UPLO::CblasUpper,
      TransA == CBLAS_TRANSPOSE::CblasTrans, Diag == CBLAS_DIAG::CblasUnit);
  auto X = AlignedBuffer<float>(N, incX);
  auto A = AlignedBuffer<float>( (N*(N + 1))/2, 1);

  auto X_expected = X;
  auto A_expected = A;

  exo_stpmv(order, Uplo, TransA, Diag, N, A.data(), X.data(), incX);
  cblas_stpmv(order, Uplo, TransA, Diag, N, A_expected.data(), X_expected.data(), incX);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 1000.f)) {
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             X_expected[i], X[i]);
      exit(1);
    }
  }

  printf("Passed!\n");
}

int main() {
  std::vector<int> N{4, 10, 1000};
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasLower};
  std::vector<CBLAS_TRANSPOSE> transA_vals{CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans};
  std::vector<CBLAS_DIAG> Diag_vals{CBLAS_DIAG::CblasUnit, CBLAS_DIAG::CblasNonUnit};
  std::vector<int> incX_vals{-2, 1, 2};

  for (auto n : N) {
    for (auto Uplo : Uplo_vals) {
      for (auto transA : transA_vals) {
        for (auto Diag : Diag_vals) {
          for (auto incX : incX_vals) {
            test_stpmv(CBLAS_ORDER::CblasRowMajor, Uplo, transA, Diag, n, incX);
          }
        }
      }
    }
  }
}
