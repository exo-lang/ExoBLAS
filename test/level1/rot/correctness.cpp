#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_rot_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(rot);

template <typename T>
void test_drot(int N, int incX, int incY, T c, T s) {
  auto X = AlignedBuffer<T>(N, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto X_expected = X;
  auto Y_expected = Y;

  rot<Exo, T>(N, X.data(), incX, Y.data(), incY, c, s);
  rot<Cblas, T>(N, X_expected.data(), incX, Y_expected.data(), incY, c, s);

  if (!X.check_buffer_equal(X_expected) || !Y.check_buffer_equal(Y_expected)) {
    failed<T>("rot", "N", N, "incX", incX, "incY", incY, "c", c, "s", s);
  }
}

template <typename T>
void run() {
  std::vector<int> N{2, 321, 1000};
  std::vector<std::tuple<T, T>> consts = {{2, 3}};
  std::vector<std::tuple<int, int, T, T>> params{{2, 2, 1, 2}};

  for (auto n : N) {
    for (auto cp : consts) {
      test_drot(n, 1, 1, std::get<0>(cp), std::get<1>(cp));
    }
  }

  for (auto n : N) {
    for (auto param : params) {
      test_drot(n, std::get<0>(param), std::get<1>(param), std::get<2>(param),
                std::get<3>(param));
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
