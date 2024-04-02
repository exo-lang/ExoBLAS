#include <cblas.h>

#include <vector>

#include "correctness_helpers.h"
#include "exo_trsv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(trsv);
generate_wrapper(trmv);

template <typename T>
void test_trsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
               const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
               const int N, const int lda, const int incX) {
  auto X = AlignedBuffer<T>(N, incX);
  auto A = AlignedBuffer2D<T>(N, lda);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) A[i * lda + j] = 2.0 + j / N;
  }
  for (int i = 0; i < X.size(); ++i) {
    X[i] = i;
  }

  auto X_expected = X;
  auto A_expected = A;

  trsv<Exo, T>(order, Uplo, TransA, Diag, N, A.data(), lda, X.data(), incX);
  trsv<Cblas, T>(order, Uplo, TransA, Diag, N, A_expected.data(), lda,
                 X_expected.data(), incX);

  if (!X.check_buffer_equal(X_expected)) {
    failed<T>("trsv", "order", order, "Uplo", Uplo, "TransA", TransA, "Diag",
              Diag, "N", N, "lda", lda, "incX", incX);
  }
}

template <typename T>
void run() {
  std::vector<CBLAS_ORDER> order_values{CblasRowMajor};
  std::vector<CBLAS_UPLO> Uplo_values{CblasUpper, CblasLower};
  std::vector<CBLAS_TRANSPOSE> trans_values{CblasNoTrans, CblasTrans};
  std::vector<CBLAS_DIAG> diag_values{CblasUnit, CblasNonUnit};
  std::vector<int> N{2, 100, 321};
  std::vector<int> incX_vals{1, 2};
  std::vector<int> lda_diffs{0, 5};
  for (auto order : order_values)
    for (auto Uplo : Uplo_values)
      for (auto TransA : trans_values)
        for (auto Diag : diag_values)
          for (auto n : N)
            for (auto incX : incX_vals)
              for (auto lda_diff : lda_diffs) {
                int lda = n + lda_diff;
                test_trsv<T>(order, Uplo, TransA, Diag, n, lda, incX);
              }
}

int main() {
  run<float>();
  run<double>();
}
