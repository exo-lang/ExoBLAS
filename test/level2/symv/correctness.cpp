#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_symv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(symv);

template <typename T>
void test_symv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const int N, const T alpha, const int lda, const int incX,
               const T beta, const int incY) {
  auto A = AlignedBuffer2D<T>(N, lda);
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);

  auto A_expected = A;
  auto X_expected = X;
  auto Y_expected = Y;

  symv<Exo, T>(order, Uplo, N, alpha, A.data(), lda, X.data(), incX, beta,
               Y.data(), incY);
  symv<Cblas, T>(order, Uplo, N, alpha, A_expected.data(), lda,
                 X_expected.data(), incX, beta, Y_expected.data(), incY);

  if (!Y.check_buffer_equal(Y_expected)) {
    failed<T>("symv", "N", N, "alpha", alpha, "lda", lda, "incX", incX, "beta",
              beta, "incY", incY, "isUpper", Uplo == CBLAS_UPLO::CblasUpper);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_UPLO> Uplo_vals{CBLAS_UPLO::CblasUpper,
                                    CBLAS_UPLO::CblasLower};
  std::vector<int> N{2, 100, 321};
  std::vector<T> alpha_vals{1, 2.4};
  std::vector<int> lda_diffs{0, 3};
  std::vector<int> incX_vals{1, 3};
  std::vector<T> beta_vals{0, 2.4};
  std::vector<int> incY_vals{1, -2};

  for (auto Uplo : Uplo_vals) {
    for (auto n : N) {
      for (auto alpha : alpha_vals) {
        for (auto lda_diff : lda_diffs) {
          int lda = n + lda_diff;
          for (auto incX : incX_vals) {
            for (auto beta : beta_vals) {
              for (auto incY : incY_vals) {
                test_symv(CBLAS_ORDER::CblasRowMajor, Uplo, n, alpha, lda, incX,
                          beta, incY);
              }
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
