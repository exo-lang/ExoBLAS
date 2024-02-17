#include <cblas.h>

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

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<T> alpha = {1.2, 0.0, 1.0};
  std::vector<std::tuple<T, int, int>> params{{1.2, 2, 2}};

  for (auto n : N) {
    for (auto a : alpha) {
      test_axpy<T>(n, a, 1, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_axpy<T>(n, std::get<0>(i), std::get<1>(i), std::get<2>(i));
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
