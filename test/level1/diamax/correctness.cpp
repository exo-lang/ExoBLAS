#include <cblas.h>

#include <iostream>
#include <vector>

#include "correctness_helpers.h"
#include "exo_idamax.h"
#include "generate_buffer.h"

void test_idamax(int N, int incX) {
  printf("Running idamax test: N = %d, incX = %d\n", N, incX);
  auto X = AlignedBuffer<double>(N, incX);
  auto X_expected = X;

  auto result = exo_idamax(N, X.data(), incX);
  auto expected = cblas_idamax(N, X_expected.data(), incX);

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
    test_idamax(n, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_idamax(n, i);
    }
  }
}
