#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_stbmv.h"
#include "generate_buffer.h"

static void BM_cblas_stbmv(benchmark::State &state) {
  const int N = state.range(0);
  const int K = state.range(1);
  const CBLAS_ORDER order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor
                                                : CBLAS_ORDER::CblasColMajor;
  const CBLAS_UPLO Uplo =
      state.range(3) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
  const CBLAS_TRANSPOSE TransA = state.range(4) == 0
                                     ? CBLAS_TRANSPOSE::CblasNoTrans
                                     : CBLAS_TRANSPOSE::CblasTrans;
  const CBLAS_DIAG Diag =
      state.range(5) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
  const int lda = N + state.range(6);
  const int incX = state.range(7);
  const int alignmentX = state.range(8);
  const int alignmentA = state.range(9);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    cblas_stbmv(order, Uplo, TransA, Diag, N, K, A.data(), lda, X.data(), incX);
  }
}

static void BM_exo_stbmv(benchmark::State &state) {
  const int N = state.range(0);
  const int K = state.range(1);
  const CBLAS_ORDER order = state.range(2) == 0 ? CBLAS_ORDER::CblasRowMajor
                                                : CBLAS_ORDER::CblasColMajor;
  const CBLAS_UPLO Uplo =
      state.range(3) == 0 ? CBLAS_UPLO::CblasLower : CBLAS_UPLO::CblasUpper;
  const CBLAS_TRANSPOSE TransA = state.range(4) == 0
                                     ? CBLAS_TRANSPOSE::CblasNoTrans
                                     : CBLAS_TRANSPOSE::CblasTrans;
  const CBLAS_DIAG Diag =
      state.range(5) == 0 ? CBLAS_DIAG::CblasNonUnit : CBLAS_DIAG::CblasUnit;
  const int lda = N + state.range(6);
  const int incX = state.range(7);
  const int alignmentX = state.range(8);
  const int alignmentA = state.range(9);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    exo_stbmv(order, Uplo, TransA, Diag, N, K, A.data(), lda, X.data(), incX);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int order = 0; order < 1; ++order) {
    for (int Uplo = 0; Uplo <= 0; ++Uplo) {
      for (int TransA = 0; TransA <= 0; ++TransA) {
        for (int Diag = 0; Diag <= 0; ++Diag) {
          for (int lda_diff = 0; lda_diff <= 0; ++lda_diff) {
            for (int incX = 1; incX <= 1; ++incX) {
              for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                  for (int N = 1; N <= (1 << 12); N *= 2) {
                    for (int K = N / 4; K >= 4; K /= 4) {
                      int lda = K + 1 + lda_diff;
                      b->Args({N, K, order, Uplo, TransA, Diag, lda, incX,
                               alignmentX, alignmentA});
                    }
                    // b -> Args({N, 0, order, Uplo, TransA, Diag, 1 + lda_diff,
                    // incX, alignmentX, alignmentA});
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

BENCHMARK(BM_cblas_stbmv)
    ->ArgNames({"n", "k", "order", "Uplo", "TransA", "Diag", "lda", "incX",
                "alignmentX", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_stbmv)
    ->ArgNames({"n", "k", "order", "Uplo", "TransA", "Diag", "lda", "incX",
                "alignmentX", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
