#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_axpy_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(axpy);

template <typename T>
void test_axpy(int N, T alpha, int incX, int incY) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  axpy<Exo, T>(N, alpha, X.data(), incX, Y.data(), incY);
  axpy<Cblas, T>(N, alpha, X_expected.data(), incX, Y_expected.data(), incY);

  if (!Y.check_buffer_equal(Y_expected)) {
    failed<T>("axpy", "N", N, "alpha", alpha, "incX", incX, "incY", incY);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<double> alpha = {1.2, 0.0, 1.0};
  std::vector<std::tuple<double, int, int>> params{{1.2, 2, 2}};

  for (auto n : N) {
    for (auto a : alpha) {
      test_axpy<float>(n, a, 1, 1);
      test_axpy<double>(n, a, 1, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_axpy<float>(n, std::get<0>(i), std::get<1>(i), std::get<2>(i));
      test_axpy<double>(n, std::get<0>(i), std::get<1>(i), std::get<2>(i));
    }
  }
}
