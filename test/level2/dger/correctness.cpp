#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dger.h"
#include "generate_buffer.h"

void test_dger(int M, int N, double alpha, int incX, int incY, int lda) {
  printf(
      "Running dger test: M = %d, N = %d, alpha = %f, incX = %d, incY = %d, "
      "lda = %d\n",
      M, N, alpha, incX, incY, lda);
  auto X = AlignedBuffer<double>(M, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto A = AlignedBuffer<double>(M * lda, 1);
  auto X_expected = X;
  auto Y_expected = Y;
  auto A_expected = A;

  exo_dger(M, N, alpha, X.data(), incX, Y.data(), incY, A.data(), lda);
  cblas_dger(CBLAS_ORDER::CblasRowMajor, M, N, alpha, X_expected.data(), incX,
             Y_expected.data(), incY, A_expected.data(), lda);

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
  std::vector<int> M{2, 100, 321};
  std::vector<int> N{3, 11, 132};
  std::vector<double> alpha = {1.2, 1.0};
  std::vector<std::tuple<double, int, int>> params{{1, 4, 5}, {4.5, -2, -4}};

  for (auto m : M) {
    for (auto n : N) {
      for (auto a : alpha) {
        test_dger(m, n, a, 1, 1, n);
      }
    }
  }

  for (auto m : M) {
    for (auto n : N) {
      for (auto i : params) {
        test_dger(m, n, std::get<0>(i), std::get<1>(i), std::get<2>(i), n);
      }
    }
  }
}
