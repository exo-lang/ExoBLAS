#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_symm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(symm);

template <typename T>
void test_symm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
               const enum CBLAS_UPLO Uplo, const int M, const int N,
               const float alpha, const int lda_diff, const int ldb_diff,
               const float beta, const int ldc_diff) {
  int lda = Side == CBLAS_SIDE::CblasLeft ? M + lda_diff : N + lda_diff;
  int ka = Side == CBLAS_SIDE::CblasLeft ? M : N;
  int ldb = N + ldb_diff;
  int ldc = N + ldc_diff;
  auto A = AlignedBuffer2D<T>(ka, lda);
  auto B = AlignedBuffer2D<T>(M, ldb);
  auto C = AlignedBuffer2D<T>(M, ldc);

  auto A_expected = A;
  auto B_expected = B;
  auto C_expected = C;

  symm<Exo, T>(Order, Side, Uplo, M, N, alpha, A.data(), lda, B.data(), ldb,
               beta, C.data(), ldc);
  symm<Cblas, T>(Order, Side, Uplo, M, N, alpha, A_expected.data(), lda,
                 B_expected.data(), ldb, beta, C_expected.data(), ldc);

  if (!C.check_buffer_equal(C_expected)) {
    failed<T>("symm", "Order", Order, "Side", Side, "Uplo", Uplo, "M", M, "N",
              N, "alpha", alpha, "lda", lda, "ldb", ldb, "beta", beta, "ldc",
              ldc);
  }
}

template <typename T>
void run() {
  std::vector<int> dims{1, 7, 32, 64, 257};
  std::vector<CBLAS_ORDER> order{CblasRowMajor};
  std::vector<CBLAS_SIDE> side{CblasLeft, CblasRight};
  std::vector<CBLAS_UPLO> uplo{CblasLower, CblasUpper};
  std::vector<int> ld_diffs{0, 5};
  std::vector<T> alphas{13.0};
  std::vector<T> betas{18.0};
  for (const auto Order : order)
    for (const auto Side : side)
      for (const auto Uplo : uplo)
        for (const auto M : dims)
          for (const auto N : dims)
            for (const auto alpha : alphas)
              for (const auto lda_diff : ld_diffs)
                for (const auto ldb_diff : ld_diffs)
                  for (const auto beta : betas)
                    for (const auto ldc_diff : ld_diffs) {
                      test_symm<T>(Order, Side, Uplo, M, N, alpha, lda_diff,
                                   ldb_diff, beta, ldc_diff);
                    }
}

int main() {
  run<float>();
  run<double>();
}
