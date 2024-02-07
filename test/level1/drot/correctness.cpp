#include <cblas.h>
#include <math.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_drot.h"
#include "generate_buffer.h"

void test_drot(int N, int incX, int incY, double c, double s) {
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  exo_drot(N, X.data(), incX, Y.data(), incY, c, s);
  cblas_drot(N, X_expected.data(), incX, Y_expected.data(), incY, c, s);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
      printf(
          "Running drot test: N = %d, incX = %d, incY = %d, c = %f, s = %f\n",
          N, incX, incY, c, s);
      printf("Failed ! expected X[%d] = %f, got %f\n", i, X_expected[i], X[i]);
      exit(1);
    }
  }

  for (int i = 0; i < Y.size(); ++i) {
    if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
      printf(
          "Running drot test: N = %d, incX = %d, incY = %d, c = %f, s = %f\n",
          N, incX, incY, c, s);
      printf("Failed ! expected Y[%d] = %f, got %f\n", i, Y_expected[i], Y[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::tuple<double, double>> consts = {{2, 3}};
  std::vector<std::tuple<int, int, double, double>> params{{2, 2, 1, 2}};

  for (auto n : N) {
    for (auto cp : consts) {
      test_drot(n, 1, 1, std::get<0>(cp), std::get<1>(cp));
    }
  }

  for (auto n : N) {
    for (auto param : params) {
      test_drot(n, std::get<0>(param), std::get<1>(param), std::get<2>(param),
                std::get<3>(param));
    }
  }
}
