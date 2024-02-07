#include <cblas.h>

#include <vector>

#include "exo_dcopy.h"
#include "generate_buffer.h"

void test_dcopy(int N, int incX, int incY) {
  auto X = AlignedBuffer<double>(N, incX);
  auto Y = AlignedBuffer<double>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  exo_dcopy(N, X.data(), incX, Y.data(), incY);
  cblas_dcopy(N, X_expected.data(), incX, Y_expected.data(), incY);

  for (int i = 0; i < Y_expected.size(); ++i) {
    if (Y_expected[i] != Y[i]) {
      printf("Running dcopy test: N = %d, incX = %d, incY = %d\n", N, incX,
             incY);
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             Y_expected[i], Y[i]);
      exit(1);
    }
  }
}

int main() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::pair<int, int>> inc{{2, 2}};

  for (auto n : N) {
    test_dcopy(n, 1, 1);
  }

  for (auto n : N) {
    for (auto i : inc) {
      test_dcopy(n, i.first, i.second);
    }
  }
}
