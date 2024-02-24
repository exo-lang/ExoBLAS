#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_gemv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(gemv);

template <typename T>
void test_gemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
               const int M, const int N, const T alpha, const int lda,
               const int incX, const T beta, const int incY) {
  auto A = AlignedBuffer2D<T>(M, lda);

  int sizeX = N;
  int sizeY = M;
  if (TransA == CBLAS_TRANSPOSE::CblasTrans) {
    sizeX = M;
    sizeY = N;
  }

  auto X = AlignedBuffer<T>(sizeX, incX);
  auto Y = AlignedBuffer<T>(sizeY, incY);

  auto A_expected = A;
  auto X_expected = X;
  auto Y_expected = Y;

  gemv<Exo, T>(order, TransA, M, N, alpha, A.data(), lda, X.data(), incX, beta,
               Y.data(), incY);
  gemv<Cblas, T>(order, TransA, M, N, alpha, A_expected.data(), lda,
                 X_expected.data(), incX, beta, Y_expected.data(), incY);

  if (!Y.check_buffer_equal(Y_expected)) {
    failed<T>("gemv", "M", M, "N", N, "alpha", alpha, "lda", lda, "incX", incX,
              "beta", beta, "incY", incY, "isTransA",
              TransA == CBLAS_TRANSPOSE::CblasTrans);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_TRANSPOSE> TransA_vals{
      CBLAS_TRANSPOSE::CblasNoTrans,
      CBLAS_TRANSPOSE::CblasTrans,
  };
  std::vector<int> N{41, 80, 101, 1024};
  std::vector<int> M{77, 103, 500};
  std::vector<T> alpha_vals{2.4};
  std::vector<int> lda_diffs{0, 5};
  std::vector<int> incX_vals{3, -2};
  std::vector<T> beta_vals{2.4};
  std::vector<int> incY_vals{5, 3};

  for (auto TransA : TransA_vals) {
    for (auto n : N) {
      for (auto m : M) {
        for (auto alpha : alpha_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            for (auto incX : incX_vals) {
              for (auto beta : beta_vals) {
                for (auto incY : incY_vals) {
                  test_gemv<T>(CBLAS_ORDER::CblasRowMajor, TransA, m, n, alpha,
                               lda, incX, beta, incY);
                }
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
