#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_saxpy.h"
#include "generate_buffer.h"

void test_saxpy(int N, float alpha, int incX, int incY) {
  auto X = AlignedBuffer<float>(N, incX);
  auto Y = AlignedBuffer<float>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  exo_saxpy(N, alpha, X.data(), incX, Y.data(), incY);
  cblas_saxpy(N, alpha, X_expected.data(), incX, Y_expected.data(), incY);

  for (int i = 0; i < Y.size(); ++i) {
    if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
      printf("Running saxpy test: N = %d, alpha = %f, incX = %d, incY = %d\n",
             N, alpha, incX, incY);
      printf("Failed ! i = %d, expected %f, got %f\n", i, Y_expected[i], Y[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<float> alpha = {-2.2, 0.0, 1.0};
  std::vector<std::tuple<float, int, int>> params{{1.2, 2, 2}, {4.5, -2, -4}};

  for (auto n : N) {
    for (auto a : alpha) {
      test_saxpy(n, a, 1, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_saxpy(n, std::get<0>(i), std::get<1>(i), std::get<2>(i));
    }
  }
}
