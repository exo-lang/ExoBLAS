#include <cblas.h>
#include <math.h>

#include <vector>

#include "blue_algorithm.h"
#include "correctness_helpers.h"
#include "exo_dnrm2.h"
#include "generate_buffer.h"

void test_dnrm2(int N, int incX, double scale) {
  printf("Running dnrm2 test: N = %d, incX = %d\n", N, incX);
  auto X = AlignedBuffer<double>(N, incX);

  for (int i = 0; i < N; i++) {
    X[i] = X[i] * scale;
  }

  auto X_expected = X;

  auto result = exo_dnrm2(N, X.data(), incX);
  auto expected = cblas_dnrm2(N, X_expected.data(), incX);
  auto epsilon = 1.f / 10.f;

  if (!check_relative_error_okay(result, expected, epsilon)) {
    printf("Failed! Expected %f, got %f\n", expected, result);
    exit(1);
  }
  printf("Passed!\n");
}

int main() {
  std::vector<int> N{1, 2, 8, 100, 64 * 64 * 64, 1000000};
  std::vector<int> inc{1, 2, 3, 5};
  std::vector<double> scale{blue_algorithm<double>::t_sml, 1.0,
                            blue_algorithm<double>::t_big * 2};

  for (auto i : inc) {
    for (auto n : N) {
      for (auto s : scale) test_dnrm2(n, i, s);
    }
  }
}
