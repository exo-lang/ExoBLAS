#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_ger_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(ger);

template <typename T>
void test_ger(const enum CBLAS_ORDER Order, int M, int N, T alpha, int incX,
              int incY, int lda) {
  auto X = AlignedBuffer<T>(M, incX);
  auto Y = AlignedBuffer<T>(N, incY);
  auto A = AlignedBuffer2D<T>(M, lda);
  auto X_expected = X;
  auto Y_expected = Y;
  auto A_expected = A;

  ger<Exo, T>(Order, M, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
              lda);
  ger<Cblas, T>(Order, M, N, alpha, X_expected.data(), incX, Y_expected.data(),
                incY, A_expected.data(), lda);

  if (!A.check_buffer_equal(A_expected)) {
    failed<T>("ger", "Order", Order, "M", M, "N", N, "alpha", alpha, "incX",
              incX, "incY", incY, "lda", lda);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_ORDER> Order_vals{CblasRowMajor};
  std::vector<int> M{77, 103, 500};
  std::vector<int> N{41, 80, 101, 1024};

  std::vector<T> alpha_vals{2.4};
  std::vector<int> lda_diffs{0, 5};
  std::vector<int> incX_vals{1, 2};
  std::vector<int> incY_vals{1, -3};

  for (auto Order : Order_vals) {
    for (auto m : M) {
      for (auto n : N) {
        for (auto alpha : alpha_vals) {
          for (auto lda_diff : lda_diffs) {
            int lda = n + lda_diff;
            for (auto incX : incX_vals) {
              for (auto incY : incY_vals) {
                test_ger<T>(Order, m, n, alpha, incX, incY, lda);
              }
            }
          }
        }
      }
    }
  }
}

int main() {
  run<float>();
  run<double>();
}
