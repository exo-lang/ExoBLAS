#include <cblas.h>
#include <math.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_srotm.h"
#include "generate_buffer.h"

void test_srotm(int N, int incX, int incY, float HFlag) {
  auto X = AlignedBuffer<float>(N, incX);
  auto Y = AlignedBuffer<float>(N, incY);
  float H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};
  auto X_expected = X;
  auto Y_expected = Y;
  float H_expected[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

  exo_srotm(N, X.data(), incX, Y.data(), incY, H);
  cblas_srotm(N, X_expected.data(), incX, Y_expected.data(), incY, H_expected);

  for (int i = 0; i < X.size(); ++i) {
    if (!check_relative_error_okay(X[i], X_expected[i], 1.f / 10000.f)) {
      printf("Running srotm test: N = %d, incX = %d, incY = %d, HFlag = %f\n",
             N, incX, incY, HFlag);
      printf("Failed ! expected X[%d] = %f, got %f\n", i, X_expected[i], X[i]);
      exit(1);
    }
  }

  for (int i = 0; i < Y.size(); ++i) {
    if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 10000.f)) {
      printf("Running srotm test: N = %d, incX = %d, incY = %d, HFlag = %f\n",
             N, incX, incY, HFlag);
      printf("Failed ! expected Y[%d] = %f, got %f\n", i, Y_expected[i], Y[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::tuple<int, int>> params = {{2, 2}, {-2, -4}};
  float HFlag[4] = {-2.0, -1.0, 0.0, 1.0};

  for (int i = 0; i < 4; ++i) {
    for (auto n : N) {
      test_srotm(n, 1, 1, HFlag[i]);
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (auto n : N) {
      for (auto p : params) {
        test_srotm(n, std::get<0>(p), std::get<1>(p), HFlag[i]);
      }
    }
  }
}
