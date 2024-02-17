#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_asum_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper_ret(asum);

template <typename T>
void test_asum(int N, int incX) {
  auto X = AlignedBuffer<T>(N, incX);
  auto X_expected = X;

  T result;
  T expected;

  result = asum<Exo, T>(N, X.data(), incX);
  expected = asum<Cblas, T>(N, X_expected.data(), incX);

  if (!check_relative_error_okay(result, expected)) {
    failed("N", N, "incX", incX);
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<int> inc{1, 3};

  for (auto n : N) {
    for (auto i : inc) {
      test_asum<float>(n, i);
      test_asum<double>(n, i);
    }
  }
}
