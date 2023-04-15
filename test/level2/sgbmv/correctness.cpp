#include <cblas.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "correctness_helpers.h"
#include "exo_sgbmv.h"
#include "generate_buffer.h"

void test_sgbmv(int M, int N, int KL, int KU, float alpha, float beta, int lda,
                int incX, int incY) {
  printf(
      "Running sgbmv test: M = %d, N = %d, KL = %d, KU = %d, alpha = %f, lda = "
      "%d, incX = %d, beta = %f, incY = %d\n",
      M, N, KL, KU, alpha, lda, incX, beta, incY);
  auto X = AlignedBuffer<float>(N, incX);
  auto Y = AlignedBuffer<float>(M, incY);
  auto A = AlignedBuffer<float>(M * lda, 1);  
  auto X_expected = X;
  auto Y_expected = Y;
  auto A_expected = A;

  exo_sgbmv(M, N, KL, KU, alpha, A.data(), lda, X.data(), incX, beta, Y.data(),
            incY);
  cblas_sgbmv(CblasRowMajor, CblasNoTrans, M, N, KL, KU, alpha,
              A_expected.data(), lda, X_expected.data(), incX, beta,
              Y_expected.data(), incY);

  for (int i = 0; i < Y.size(); ++i) {
    if (!check_relative_error_okay(Y[i], Y_expected[i], 1.f / 1000.f)) {    
      printf("Failed ! mem offset = %d, expected %f, got %f\n", i,
             Y_expected[i], Y[i]);
      
      exit(1);
    }
  }

  printf("Passed!\n");
}

int main() {
  std::vector<std::tuple<int, int, int, int> > sizes{
      {2, 2, 1, 1}, {15, 20, 0, 5}, {15, 20, 5, 0}, {20, 15, 5, 0}, {256, 256, 255, 255}
      , {8192, 8192, 8191, 8191}
      };

  std::vector<std::tuple<float, float> > consts{
      {1.0, 1.0}, {1.0, 1.2}, {1.2, 1.0}, {1.2, 1.2}};

  for (auto s : sizes) {
    for (auto c : consts) {
      test_sgbmv(std::get<0>(s), std::get<1>(s), std::get<2>(s), std::get<3>(s),
                 std::get<0>(c), std::get<1>(c),
                 std::get<2>(s) + std::get<3>(s) + 1, 1, 1);
    }
  }
}
