#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_syr2_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr2);

template <typename T>
void test_syr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, int N,
               T alpha, int incX, int incY, int lda) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto A = AlignedBuffer2D<T>(N, lda);
  auto X_expected = X;
  auto Y_expected = Y;
  auto A_expected = A;

  syr2<Exo, T>(CBLAS_ORDER::CblasRowMajor, Uplo, N, alpha, X.data(), incX,
               Y.data(), incY, A.data(), lda);
  syr2<Cblas, T>(CBLAS_ORDER::CblasRowMajor, Uplo, N, alpha, X_expected.data(),
                 incX, Y_expected.data(), incY, A_expected.data(), lda);

  if (!A.check_buffer_equal(A_expected)) {
    failed<T>("syr2", "N", N, "alpha", alpha, "lda", lda, "incX", incX, "incY",
              incY, "isUpper", Uplo == CBLAS_UPLO::CblasUpper);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<int> N{2, 100, 321};
  std::vector<T> alpha_vals{1.2};
  std::vector<int> incX_vals{1, 2};
  std::vector<int> incY_vals{1, -3};
  std::vector<int> lda_diffs{0, 5};

  for (auto Uplo : Uplo_vals) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto incX : incX_vals) {
          for (auto incY : incY_vals) {
            for (auto lda_diff : lda_diffs) {
              int lda = n + lda_diff;
              test_syr2<T>(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, incX,
                           incY, lda);
            }
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
