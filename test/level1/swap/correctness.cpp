#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_swap_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(swap);

template <typename T>
void test_swap(int N, int incX, int incY) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  swap<Exo, T>(N, X.data(), incX, Y.data(), incY);
  swap<Cblas, T>(N, X_expected.data(), incX, Y_expected.data(), incY);

  if (!Y.check_buffer_equal(Y_expected)) {
    failed<T>("swap", "N", N, "incX", incX, "incY", incY);
  }
}

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}};

  for (auto n : N) {
    test_swap<T>(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_swap<T>(n, i.first, i.second);
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
