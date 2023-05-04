#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_ssyr2.h"
#include "generate_buffer.h"

static void BM_cblas_ssyr2(benchmark::State &state) {
  int N = state.range(0);
  CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor
                                          : CBLAS_ORDER::CblasColMajor;
  CBLAS_UPLO Uplo =
      state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  float alpha = state.range(3);
  int incX = state.range(4);
  int incY = state.range(5);
  int lda = N + state.range(6);
  size_t alignmentX = state.range(7);
  size_t alignmentY = state.range(8);
  size_t alignmentA = state.range(9);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);
  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    cblas_ssyr2(order, Uplo, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
                lda);
  }
}

static void BM_exo_ssyr2(benchmark::State &state) {
  int N = state.range(0);
  CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor
                                          : CBLAS_ORDER::CblasColMajor;
  CBLAS_UPLO Uplo =
      state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  float alpha = state.range(3);
  int incX = state.range(4);
  int incY = state.range(5);
  int lda = N + state.range(6);
  size_t alignmentX = state.range(7);
  size_t alignmentY = state.range(8);
  size_t alignmentA = state.range(9);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);
  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    exo_ssyr2(order, Uplo, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
              lda);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int order = 0; order < 1; ++order) {
    for (int Uplo = 0; Uplo <= 1; ++Uplo) {
      for (int alpha = 3; alpha <= 3; ++alpha) {
        for (int lda_diff = 0; lda_diff < 1; ++lda_diff) {
          for (int incX = 1; incX <= 1; ++incX) {
            for (int incY = 1; incY <= 1; ++incY) {
              for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                  for (int alignmentY = 64; alignmentY <= 64; ++alignmentY) {
                    for (int N = 1; N <= (1 << 13); N *= 2) {
                      b->Args({N, order, Uplo, alpha, incX, incY, lda_diff,
                               alignmentX, alignmentY, alignmentA});
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

BENCHMARK(BM_cblas_ssyr2)
    ->ArgNames({"n", "order", "Uplo", "alpha", "incX", "incY", "lda_diff",
                "alignmentX", "alignmentY", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_ssyr2)
    ->ArgNames({"n", "order", "Uplo", "alpha", "incX", "incY", "lda_diff",
                "alignmentX", "alignmentY", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
