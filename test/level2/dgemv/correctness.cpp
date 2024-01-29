#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dgemv.h"
#include "generate_buffer.h"

void test_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N, const double alpha, const int lda,
                const int incX, const double beta, const int incY) {
  printf(
      "Running dgemv test: M = %d, N = %d, alpha = %f, lda = %d, incX = %d, "
      "beta = %f, "
      "incY = %d, isTransA = %d\n",
      M, N, alpha, lda, incX, beta, incY,
      TransA == CBLAS_TRANSPOSE::CblasTrans);

  auto A = AlignedBuffer<double>(M * lda, 1);

  int sizeX = N;
  int sizeY = M;
  if (TransA == CBLAS_TRANSPOSE::CblasTrans) {
    sizeX = M;
    sizeY = N;
  }

  auto X = AlignedBuffer<double>(sizeX, incX);
  auto Y = AlignedBuffer<double>(sizeY, incY);

  auto A_expected = A;
  auto X_expected = X;
  auto Y_expected = Y;

  exo_dgemv(order, TransA, M, N, alpha, A.data(), lda, X.data(), incX, beta,
            Y.data(), incY);
  cblas_dgemv(order, TransA, M, N, alpha, A_expected.data(), lda,
              X_expected.data(), incX, beta, Y_expected.data(), incY);

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
  std::vector<CBLAS_TRANSPOSE> TransA_vals{CBLAS_TRANSPOSE::CblasTrans,
                                           CBLAS_TRANSPOSE::CblasNoTrans};
  std::vector<int> N{41, 80, 101, 1024};
  std::vector<int> M{77, 103, 500};
  std::vector<double> alpha_vals{2.4};
  std::vector<int> lda_diffs{0, 5};
  std::vector<int> incX_vals{1, -2};
  std::vector<double> beta_vals{2.4};
  std::vector<int> incY_vals{1, 3};

  for (auto TransA : TransA_vals) {
    for (auto n : N) {
      for (auto m : M) {
        for (auto alpha : alpha_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            for (auto incX : incX_vals) {
              for (auto beta : beta_vals) {
                for (auto incY : incY_vals) {
                  test_dgemv(CBLAS_ORDER::CblasRowMajor, TransA, m, n, alpha,
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
