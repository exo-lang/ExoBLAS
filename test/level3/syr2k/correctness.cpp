#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_syr2k_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr2k);

template <typename T>
void test_syr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                const T alpha, const int lda_diff, const int ldb_diff,
                const T beta, const int ldc_diff) {
  auto A_dims = get_dims(Trans, N, K, lda_diff);
  int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second);
  auto B_dims = get_dims(Trans, N, K, ldb_diff);
  int ldb = B_dims.second;
  auto B = AlignedBuffer2D<T>(B_dims.first, B_dims.second);
  int ldc = N + ldc_diff;
  auto C = AlignedBuffer2D<T>(N, ldc);

  auto A_expected = A;
  auto B_expected = B;
  auto C_expected = C;

  syr2k<Exo, T>(Order, Uplo, Trans, N, K, alpha, A.data(), lda, B.data(), ldb,
                beta, C.data(), ldc);
  syr2k<Cblas, T>(Order, Uplo, Trans, N, K, alpha, A_expected.data(), lda,
                  B_expected.data(), ldb, beta, C_expected.data(), ldc);

  if (!C.check_buffer_equal(C_expected)) {
    failed<T>("syr2k", "Order", Order, "Uplo", Uplo, "Trans", Trans, "N", N,
              "K", K, "alpha", alpha, "lda", lda, "ldb", ldb, "beta", beta,
              "ldc", ldc);
  }
}

template <typename T>
void run() {
  std::vector<int> dims{1, 7, 32, 64, 257};
  std::vector<CBLAS_TRANSPOSE> trans{CblasNoTrans, CblasTrans};
  std::vector<CBLAS_UPLO> uplo{CblasLower, CblasUpper};
  std::vector<int> ld_diffs{0, 5};
  std::vector<T> alphas{13.0};
  std::vector<T> betas{1.0};
  for (const auto Uplo : uplo)
    for (const auto N : dims)
      for (const auto K : dims)
        for (const auto Trans : trans)
          for (const auto lda_diff : ld_diffs)
            for (const auto ldb_diff : ld_diffs)
              for (const auto ldc_diff : ld_diffs)
                for (const auto alpha : alphas)
                  for (const auto beta : betas) {
                    test_syr2k<T>(CBLAS_ORDER::CblasRowMajor, Uplo, Trans, N, K,
                                  alpha, lda_diff, ldb_diff, beta, ldc_diff);
                  }
}

int main() {
  run<float>();
  run<double>();
}
