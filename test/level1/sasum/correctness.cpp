#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_sasum.h"
#include "generate_buffer.h"

void test_sasum(int N, int incX) {
  auto X = AlignedBuffer<float>(N, incX);
  auto X_expected = X;

  auto result = exo_sasum(N, X.data(), incX);
  auto expected = cblas_sasum(N, X_expected.data(), incX);

  auto epsilon = 1.f / 100.f;

  if (!check_relative_error_okay(result, expected, epsilon)) {
    printf("Running sasum test: N = %d, incX = %d\n", N, incX);
    printf("Failed! Expected %f, got %f\n", expected, result);
    exit(1);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<int> inc{4};

  for (auto n : N) {
    test_sasum(n, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_sasum(n, i);
    }
  }
}
