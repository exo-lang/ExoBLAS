#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_dasum.h"
#include "generate_buffer.h"

void test_dasum(int N, int incX) {
  printf("Running dasum test: N = %d, incX = %d\n", N, incX);
  auto X = AlignedBuffer<double>(N, incX);
  auto X_expected = X;

  auto result = exo_dasum(N, X.data(), incX);
  auto expected = cblas_dasum(N, X_expected.data(), incX);

  auto epsilon = 1.f / 1000.f;

  if (!check_relative_error_okay(result, expected, epsilon)) {
    printf("Failed! Expected %f, got %f\n", expected, result);
    exit(1);
  }

  printf("Passed!\n");
}

int main() {
  std::vector<int> N{1, 2, 8, 100, 64 * 64 * 64, 10000000};
  std::vector<int> inc{2, 3, 5, 10};

  for (auto n : N) {
    test_dasum(n, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dasum(n, i);
    }
  }
}
