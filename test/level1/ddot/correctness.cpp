#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_ddot.h"
#include "generate_buffer.h"

void test_ddot(int N, int incX, int incY) {
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  auto result = exo_ddot(N, X.data(), incX, Y.data(), incY);
  auto expected =
      cblas_ddot(N, X_expected.data(), incX, Y_expected.data(), incY);

  auto epsilon = 1.f / 1000.f;

  if (!check_relative_error_okay(result, expected, epsilon)) {
    printf("Running ddot test: N = %d, incX = %d, incY = %d\n", N, incX, incY);
    printf("Failed! Expected %f, got %f\n", expected, result);
    exit(1);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{3, 3}};

  for (auto n : N) {
    test_ddot(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_ddot(n, i.first, i.second);
    }
  }
}
