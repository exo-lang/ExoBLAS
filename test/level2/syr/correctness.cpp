#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_syr_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr);

template <typename T>
void test_syr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
              const int N, const T alpha, const int incX, const int lda) {
  auto X = AlignedBuffer<T>(N, incX);
  auto A = AlignedBuffer2D<T>(N, lda);
  auto X_expected = X;
  auto A_expected = A;

  syr<Exo, T>(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
  syr<Cblas, T>(order, Uplo, N, alpha, X_expected.data(), incX,
                A_expected.data(), lda);

  if (!A.check_buffer_equal(A_expected)) {
    failed<T>("syr", "N", N, "alpha", alpha, "lda", lda, "incX", incX,
              "isUpper", Uplo == CBLAS_UPLO::CblasUpper);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_UPLO> Uplo_values{CBLAS_UPLO::CblasUpper,
                                      CBLAS_UPLO::CblasLower};
  std::vector<int> N{2, 100, 321, 1115};
  std::vector<T> alpha_vals = {1.2};
  std::vector<int> incX_vals{1, 2};
  std::vector<int> lda_diffs{0, 5};

  for (auto Uplo : Uplo_values) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto incX : incX_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            test_syr<T>(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, incX, lda);
          }
        }
      }
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
