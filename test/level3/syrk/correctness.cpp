#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_syrk_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syrk);

template <typename T>
void test_syrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
               const float alpha, const int lda_diff, const float beta,
               const int ldc_diff) {
  auto A_dims = get_dims(Trans, N, K, lda_diff);
  const int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second);
  const int ldc = N + ldc_diff;
  auto C = AlignedBuffer2D<T>(N, ldc);

  auto A_expected = A;
  auto C_expected = C;

  syrk<Exo, T>(Order, Uplo, Trans, N, K, alpha, A.data(), lda, beta, C.data(),
               ldc);
  syrk<Cblas, T>(Order, Uplo, Trans, N, K, alpha, A_expected.data(), lda, beta,
                 C_expected.data(), ldc);

  if (!C.check_buffer_equal(C_expected)) {
    failed<T>("syrk", "Order", Order, "Uplo", Uplo, "Trans", Trans, "N", N, "K",
              K, "alpha", alpha, "lda", lda, "beta", beta, "ldc", ldc);
  }
}

template <typename T>
void run() {
  std::vector<int> dims{1, 7, 32, 64, 257};
  std::vector<CBLAS_TRANSPOSE> trans{CblasNoTrans};
  std::vector<CBLAS_UPLO> uplo{CblasLower, CblasUpper};
  std::vector<int> ld_diffs{0, 5};
  std::vector<T> alphas{13.0};
  std::vector<T> betas{1.0};
  for (const auto Uplo : uplo)
    for (const auto N : dims)
      for (const auto K : dims)
        for (const auto Trans : trans)
          for (const auto lda_diff : ld_diffs)
            for (const auto ldc_diff : ld_diffs)
              for (const auto alpha : alphas)
                for (const auto beta : betas) {
                  test_syrk<T>(CBLAS_ORDER::CblasRowMajor, Uplo, Trans, N, K,
                               alpha, lda_diff, beta, ldc_diff);
                }
}

int main() {
  run<float>();
  run<double>();
}
