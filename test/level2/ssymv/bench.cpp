#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_ssymv.h"
#include "generate_buffer.h"

static void BM_cblas_ssymv(benchmark::State &state) {
  const int N = state.range(0);
  const enum CBLAS_ORDER order = state.range(1) == 0
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_UPLO Uplo =
      state.range(2) == 1 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  const float alpha = state.range(3);
  const int lda_diff = state.range(4);
  int lda = N + lda_diff;
  const int incX = state.range(5);
  const float beta = state.range(6);
  const int incY = state.range(7);
  size_t alignmentA = state.range(8);
  size_t alignmentX = state.range(9);
  size_t alignmentY = state.range(10);

  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);
  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    cblas_ssymv(order, Uplo, N, alpha, A.data(), lda, X.data(), incX, beta,
                Y.data(), incY);
  }
}

static void BM_exo_ssymv(benchmark::State &state) {
  const int N = state.range(0);
  const enum CBLAS_ORDER order = state.range(1) == 0
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_UPLO Uplo =
      state.range(2) == 1 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  const float alpha = state.range(3);
  const int lda_diff = state.range(4);
  int lda = N + lda_diff;
  const int incX = state.range(5);
  const float beta = state.range(6);
  const int incY = state.range(7);
  size_t alignmentA = state.range(8);
  size_t alignmentX = state.range(9);
  size_t alignmentY = state.range(10);

  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);
  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    exo_ssymv(order, Uplo, N, alpha, A.data(), lda, X.data(), incX, beta,
              Y.data(), incY);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int order = 0; order < 1; ++order) {
    for (int Uplo = 0; Uplo <= 1; ++Uplo) {
      for (int alpha = 3; alpha <= 3; ++alpha) {
        for (int lda_diff = 0; lda_diff <= 0; ++lda_diff) {
          for (int incX = 1; incX <= 1; ++incX) {
            for (int beta = 5; beta <= 5; ++beta) {
              for (int incY = 1; incY <= 1; ++incY) {
                for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                  for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                    for (int alignmentY = 64; alignmentY <= 64; ++alignmentY) {
                      for (int N = 1; N <= (1 << 13); N *= 2) {
                        b->Args({N, order, Uplo, alpha, lda_diff, incX, beta,
                                 incY, alignmentA, alignmentX, alignmentY});
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

BENCHMARK(BM_cblas_ssymv)
    ->ArgNames({"n", "order", "Uplo", "alpha", "lda_diff", "incX", "beta",
                "incY", "alignmentA", "alignmentX", "alignmentY"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_ssymv)
    ->ArgNames({"n", "order", "Uplo", "alpha", "lda_diff", "incX", "beta",
                "incY", "alignmentA", "alignmentX", "alignmentY"})
    ->Apply(CustomArgumentsPacked);
