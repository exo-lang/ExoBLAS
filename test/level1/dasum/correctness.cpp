#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_dasum.h"
#include "generate_buffer.h"

void test_dasum(int N, int incX) {
  auto X = AlignedBuffer<double>(N, incX);
  auto X_expected = X;

  auto result = exo_dasum(N, X.data(), incX);
  auto expected = cblas_dasum(N, X_expected.data(), incX);

  auto epsilon = 1.f / 1000.f;

  if (!check_relative_error_okay(result, expected, epsilon)) {
    printf("Running dasum test: N = %d, incX = %d\n", N, incX);
    printf("Failed! Expected %f, got %f\n", expected, result);
    exit(1);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<int> inc{1, 3};

  for (auto n : N) {
    test_dasum(n, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dasum(n, i);
    }
  }
}
