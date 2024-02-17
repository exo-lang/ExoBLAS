#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_copy_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(copy);

template <typename T>
void test_copy(int N, int incX, int incY) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  copy<Exo, T>(N, X.data(), incX, Y.data(), incY);
  copy<Cblas, T>(N, X_expected.data(), incX, Y_expected.data(), incY);

  if (!Y.check_buffer_equal(Y_expected)) {
    failed<T>("copy", "N", N, "incX", incX, "incY", incY);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}};

  for (auto n : N) {
    test_copy<float>(n, 1, 1);
    test_copy<double>(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_copy<float>(n, i.first, i.second);
      test_copy<double>(n, i.first, i.second);
    }
  }
}
