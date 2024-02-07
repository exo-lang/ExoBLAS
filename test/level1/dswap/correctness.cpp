#include <cblas.h>

#include <vector>

#include "exo_dswap.h"
#include "generate_buffer.h"

void test_dswap(int N, int incX, int incY) {
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  exo_dswap(N, X.data(), incX, Y.data(), incY);
  cblas_dswap(N, X_expected.data(), incX, Y_expected.data(), incY);

  for (int i = 0; i < Y.size(); ++i) {
    if (Y[i] != Y_expected[i]) {
      printf("Running dswap test: N = %d, incX = %d, incY = %d\n", N, incX,
             incY);
      printf("Failed ! memory offset = %d, expected %f, got %f\n", i,
             Y_expected[i], Y[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}};

  for (auto n : N) {
    test_dswap(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dswap(n, i.first, i.second);
    }
  }
}
