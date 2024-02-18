#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_dsdot_wrapper.h"
#include "generate_buffer.h"

void test_dsdot(int N, int incX, int incY) {
  auto X = AlignedBuffer<float>(N, incX);
  auto Y = AlignedBuffer<float>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  auto result = exo_dsdot(N, X.data(), incX, Y.data(), incY);
  auto expected =
      cblas_dsdot(N, X_expected.data(), incX, Y_expected.data(), incY);

  if (!check_relative_error_okay(result, expected)) {
    failed<double>("sdot", "N", N, "incX", incX, "incY", incY);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}};

  for (auto n : N) {
    test_dsdot(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dsdot(n, i.first, i.second);
    }
  }
}
