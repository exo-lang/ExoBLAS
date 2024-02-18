#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_scal_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(scal);

template <typename T>
void test_scal(int N, T alpha, int incX) {
  auto X = AlignedBuffer<T>(N, incX);
  auto X_expected = X;

  scal<Exo, T>(N, alpha, X.data(), incX);
  scal<Cblas, T>(N, alpha, X_expected.data(), incX);

  if (!X.check_buffer_equal(X_expected)) {
    failed<T>("scal", "N", N, "alpha", alpha, "incX", incX);
  }
}

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<T> alphas{0, 1, 2};
  std::vector<std::tuple<T, int>> params{{1, 4}};

  for (auto n : N) {
    for (auto alpha : alphas) {
      test_scal<T>(n, alpha, 1);
    }
  }

  for (auto n : N) {
    for (auto i : params) {
      test_scal<T>(n, std::get<0>(i), std::get<1>(i));
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
