#include <cblas.h>

#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_rotm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(rotm);

template <typename T>
void test_rotm(int N, int incX, int incY, T HFlag) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  T H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};
  auto X_expected = X;
  auto Y_expected = Y;
  T H_expected[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

  rotm<Exo, T>(N, X.data(), incX, Y.data(), incY, H);
  rotm<Cblas, T>(N, X_expected.data(), incX, Y_expected.data(), incY,
                 H_expected);

  if (!X.check_buffer_equal(X_expected) || !Y.check_buffer_equal(Y_expected)) {
    failed<T>("rotm", "N", N, "incX", incX, "incY", incY, "HFlag", HFlag);
  }
}

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::tuple<int, int>> params = {{2, 2}};
  T HFlag[4] = {-2.0, -1.0, 0.0, 1.0};

  for (int i = 0; i < 4; ++i) {
    for (auto n : N) {
      test_rotm<T>(n, 1, 1, HFlag[i]);
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (auto n : N) {
      for (auto p : params) {
        test_rotm<T>(n, std::get<0>(p), std::get<1>(p), HFlag[i]);
      }
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
