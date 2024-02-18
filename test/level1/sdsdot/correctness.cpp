#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_sdsdot.h"
#include "generate_buffer.h"

void test_sdsdot(int N, float alpha, int incX, int incY) {
  auto X = AlignedBuffer<float>(N, incX);
  auto Y = AlignedBuffer<float>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  auto result = exo_sdsdot(N, alpha, X.data(), incX, Y.data(), incY);
  auto expected =
      cblas_sdsdot(N, alpha, X_expected.data(), incX, Y_expected.data(), incY);

  if (!check_relative_error_okay(result, expected)) {
    failed<float>("dsdot", "N", N, "incX", incX, "incY", incY);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}, {-2, -4}};

  for (auto n : N) {
    test_sdsdot(n, 0.0, 1, 1);
    test_sdsdot(n, -1.3, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_sdsdot(n, 0.0, i.first, i.second);
      test_sdsdot(n, -1.3, i.first, i.second);
    }
  }
}
