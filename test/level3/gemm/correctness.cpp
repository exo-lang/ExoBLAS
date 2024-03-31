#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_gemm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(gemm);

template <typename T>
void test_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
               const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
               const int K, const T alpha, const int lda_diff,
               const int ldb_diff, const T beta, const int ldc_diff) {
  auto A_dims = get_dims(TransA, M, K, lda_diff);
  const int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second);
  auto B_dims = get_dims(TransB, K, N, ldb_diff);
  const int ldb = B_dims.second;
  auto B = AlignedBuffer2D<T>(B_dims.first, B_dims.second);
  const int ldc = N + ldc_diff;
  auto C = AlignedBuffer2D<T>(M, ldc);

  auto A_expected = A;
  auto B_expected = B;
  auto C_expected = C;

  gemm<Exo, T>(Order, TransA, TransB, M, N, K, alpha, A.data(), lda, B.data(),
               ldb, beta, C.data(), ldc);

  gemm<Cblas, T>(Order, TransA, TransB, M, N, K, alpha, A_expected.data(), lda,
                 B_expected.data(), ldb, beta, C_expected.data(), ldc);

  if (!C.check_buffer_equal(C_expected)) {
    failed<T>("gemm", "Order", Order, "TransA", TransA, "TransB", TransB, "M",
              M, "N", N, "K", K, "alpha", alpha, "lda", lda, "ldb", ldb, "beta",
              beta, "ldc", ldc);
  }
}

template <typename T>
void run() {
  std::vector<int> dims{1, 7, 32, 64, 257};
  std::vector<CBLAS_TRANSPOSE> trans{CblasNoTrans, CblasTrans};
  std::vector<int> ld_diffs{0, 5};
  std::vector<T> alphas{13.0};
  std::vector<T> betas{18.0};

  for (const auto M : dims)
    for (const auto N : dims)
      for (const auto K : dims)
        for (const auto TransA : trans)
          for (const auto TransB : trans)
            for (const auto lda_diff : ld_diffs)
              for (const auto ldb_diff : ld_diffs)
                for (const auto ldc_diff : ld_diffs)
                  for (const auto alpha : alphas)
                    for (const auto beta : betas) {
                      test_gemm<T>(CBLAS_ORDER::CblasRowMajor, TransA, TransB,
                                   M, N, K, alpha, lda_diff, ldb_diff, beta,
                                   ldc_diff);
                    }
}

int main() {
  run<float>();
  run<double>();
}
