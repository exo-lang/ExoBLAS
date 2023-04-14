#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_strmm.h"
#include "generate_buffer.h"

static void BM_cblas_strmm(benchmark::State &state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const CBLAS_ORDER Order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor
                                                : CBLAS_ORDER::CblasColMajor;
  const CBLAS_SIDE Side =
      state.range(3) == 0 ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
  const CBLAS_UPLO Uplo =
      state.range(4) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
  const CBLAS_TRANSPOSE TransA = state.range(5) == 0
                                     ? CBLAS_TRANSPOSE::CblasNoTrans
                                     : CBLAS_TRANSPOSE::CblasTrans;
  const CBLAS_DIAG Diag =
      state.range(6) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
  float alpha = state.range(7);
  const int lda = state.range(8);
  const int ldb = state.range(9);
  const int alignmentA = state.range(10);
  const int alignmentB = state.range(11);

  int K = Side == CBLAS_SIDE::CblasLeft ? M : N;

  auto A = AlignedBuffer<float>(K * lda, 1, alignmentA);
  auto B = AlignedBuffer<float>(M * ldb, 1, alignmentB);

  for (auto _ : state) {
    cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A.data(), lda,
                B.data(), ldb);
  }
}

static void BM_exo_strmm(benchmark::State &state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const CBLAS_ORDER Order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor
                                                : CBLAS_ORDER::CblasColMajor;
  const CBLAS_SIDE Side =
      state.range(3) == 0 ? CBLAS_SIDE::CblasLeft : CBLAS_SIDE::CblasRight;
  const CBLAS_UPLO Uplo =
      state.range(4) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
  const CBLAS_TRANSPOSE TransA = state.range(5) == 0
                                     ? CBLAS_TRANSPOSE::CblasNoTrans
                                     : CBLAS_TRANSPOSE::CblasTrans;
  const CBLAS_DIAG Diag =
      state.range(6) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
  float alpha = state.range(7);
  const int lda = state.range(8);
  const int ldb = state.range(9);
  const int alignmentA = state.range(10);
  const int alignmentB = state.range(11);

  int K = Side == CBLAS_SIDE::CblasLeft ? M : N;

  auto A = AlignedBuffer<float>(K * lda, 1, alignmentA);
  auto B = AlignedBuffer<float>(M * ldb, 1, alignmentB);

  for (auto _ : state) {
    exo_strmm(Order, Side, Uplo, TransA, Diag, M, N, alpha, A.data(), lda,
              B.data(), ldb);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int Order = 0; Order < 1; ++Order) {
    for (int Side = 0; Side <= 1; ++Side) {
      for (int Uplo = 0; Uplo <= 1; ++Uplo) {
        for (int TransA = 0; TransA <= 1; ++TransA) {
          for (int Diag = 0; Diag <= 1; ++Diag) {
            for (int alpha = 0; alpha <= 1; ++alpha) {
              for (int lda_diff = 0; lda_diff < 1; ++lda_diff) {
                for (int ldb_diff = 0; ldb_diff < 1; ++ldb_diff) {
                  for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                    for (int alignmentB = 64; alignmentB <= 64; ++alignmentB) {
                      for (int M = 1; M <= (1 << 10); M *= 2) {
                        for (int N = M; N <= M; N *= 2) {
                          int K = Side == CBLAS_SIDE::CblasLeft ? M : N;
                          int lda = K + lda_diff;
                          int ldb = N + ldb_diff;
                          b->Args({M, N, Order, Side, Uplo, TransA, Diag, alpha,
                                   lda, ldb, alignmentA, alignmentB});
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

BENCHMARK(BM_cblas_strmm)
    ->ArgNames({"M", "N", "Order", "Side", "Uplo", "TransA", "Diag", "alpha",
                "lda", "ldb", "alignmentA", "alignmentB"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_strmm)
    ->ArgNames({"M", "N", "Order", "Side", "Uplo", "TransA", "Diag", "alpha",
                "lda", "ldb", "alignmentA", "alignmentB"})
    ->Apply(CustomArgumentsPacked);
