#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_sscal.h"
#include "generate_buffer.h"

void test_sscal(int N, float alpha, int incX) {
  auto X = AlignedBuffer<float>(N, incX);
  auto X_expected = X;

  exo_sscal(N, alpha, X.data(), incX);
  cblas_sscal(N, alpha, X_expected.data(), incX);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 100.f)) {
      printf("Running sscal test: N = %d, alpha = %f, incX = %d\n", N, alpha,
             incX);
      printf("Failed ! memory offset = %d, expected %f, got %f\n", i,
             X_expected[i], X[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<float> alphas{0, 1, 2};
  std::vector<std::tuple<float, int>> params{{1.2, 2}, {1.3, 10}};

  for (auto n : N) {
    for (auto alpha : alphas) {
      test_sscal(n, alpha, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_sscal(n, std::get<0>(i), std::get<1>(i));
    }
  }
}
