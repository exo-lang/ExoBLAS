#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_dot_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper_ret(dot);

template <typename T>
void test_dot(int N, int incX, int incY) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  auto result = dot<Exo, T>(N, X.data(), incX, Y.data(), incY);
  auto expected =
      dot<Cblas, T>(N, X_expected.data(), incX, Y_expected.data(), incY);

  if (!check_relative_error_okay(result, expected)) {
    failed<T>("dot", "N", N, "incX", incX, "incY", incY);
  }
}

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{3, 3}};

  for (auto n : N) {
    test_dot<T>(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dot<T>(n, i.first, i.second);
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
