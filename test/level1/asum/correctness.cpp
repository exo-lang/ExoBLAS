#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_asum_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

template <typename T>
void test_asum(int N, int incX) {
  auto X = AlignedBuffer<T>(N, incX);
  auto X_expected = X;

  T result;
  T expected;
#define args N, X.data(), incX

  if constexpr (std::is_same<T, float>::value) {
    result = exo_sasum(args);
    expected = cblas_sasum(args);
  } else {
    result = exo_dasum(args);
    expected = cblas_dasum(args);
  }

  if (!check_relative_error_okay(result, expected)) {
    std::cout << "Running " << kernel_name<Exo, T>("asum") << " test"
              << std::endl;
    std::cout << "Params "
              << "N = " << N << ", incX = " << incX << std::endl;
    std::cout << "Failed! Expected " << expected << ", got " << result
              << std::endl;
    exit(1);
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
