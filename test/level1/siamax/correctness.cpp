#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_isamax.h"
#include "generate_buffer.h"

void test_isamax(int N, int incX) {
  printf("Running isamax test: N = %d, incX = %d\n", N, incX);
  auto X = AlignedBuffer<float>(N, incX);
  auto X_expected = X;

  auto result = exo_isamax(N, X.data(), incX);
  auto expected = cblas_isamax(N, X_expected.data(), incX);

  if (result != expected) {
    printf("Failed! Expected %ld, got %d\n", expected, result);
    exit(1);
  }

  printf("Passed!\n");
}

int main() {
  std::vector<int> N{1, 2, 8, 100, 64 * 64 * 64, 10000000};
  std::vector<int> inc{2, 3, 5, -1, 10};

  for (auto n : N) {
    test_isamax(n, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_isamax(n, i);
    }
  }
}
