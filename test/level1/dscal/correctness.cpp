#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_dscal.h"
#include "generate_buffer.h"

void test_dscal(int N, double alpha, int incX) {
  auto X = AlignedBuffer<double>(N, incX);
  auto X_expected = X;

  exo_dscal(N, alpha, X.data(), incX);
  cblas_dscal(N, alpha, X_expected.data(), incX);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
      printf("Running dscal test: N = %d, alpha = %f, incX = %d\n", N, alpha,
             incX);
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             X_expected[i], X[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<double> alphas{0, 1, 2};
  std::vector<std::tuple<double, int>> params{{1, 4}};

  for (auto n : N) {
    for (auto alpha : alphas) {
      test_dscal(n, alpha, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_dscal(n, std::get<0>(i), std::get<1>(i));
    }
  }
}
