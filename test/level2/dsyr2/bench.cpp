#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_dsyr2.h"
#include "generate_buffer.h"

static void BM_cblas_dsyr2(benchmark::State &state) {
  int N = state.range(0);
  CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor
                                          : CBLAS_ORDER::CblasColMajor;
  CBLAS_UPLO Uplo =
      state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  double alpha = state.range(3);
  int incX = state.range(4);
  int incY = state.range(5);
  int lda = state.range(6);
  size_t alignmentX = state.range(7);
  size_t alignmentY = state.range(8);
  size_t alignmentA = state.range(9);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);
  auto Y = AlignedBuffer<double>(N, incY, alignmentY);
  auto A = AlignedBuffer<double>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    cblas_dsyr2(order, Uplo, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
                lda);
  }
}

static void BM_exo_dsyr2(benchmark::State &state) {
  int N = state.range(0);
  CBLAS_ORDER order = state.range(1) == 0 ? CBLAS_ORDER::CblasRowMajor
                                          : CBLAS_ORDER::CblasColMajor;
  CBLAS_UPLO Uplo =
      state.range(2) == 0 ? CBLAS_UPLO::CblasUpper : CBLAS_UPLO::CblasLower;
  double alpha = state.range(3);
  int incX = state.range(4);
  int incY = state.range(5);
  int lda = state.range(6);
  size_t alignmentX = state.range(7);
  size_t alignmentY = state.range(8);
  size_t alignmentA = state.range(9);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);
  auto Y = AlignedBuffer<double>(N, incY, alignmentY);
  auto A = AlignedBuffer<double>(N * lda, 1, alignmentA);

  for (auto _ : state) {
    exo_dsyr2(order, Uplo, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
              lda);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int order = 0; order < 1; ++order) {
    for (int Uplo = 0; Uplo <= 1; ++Uplo) {
      for (int alpha = 0; alpha <= 2; ++alpha) {
        for (int lda_diff = 0; lda_diff < 1; ++lda_diff) {
          for (int incX = 1; incX <= 1; ++incX) {
            for (int incY = 1; incY <= 1; ++incY) {
              for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                  for (int alignmentY = 64; alignmentY <= 64; ++alignmentY) {
                    for (int N = 1; N <= (1 << 10); N *= 2) {
                      int lda = N + lda_diff;
                      b->Args({N, order, Uplo, alpha, incX, incY, lda,
                               alignmentX, alignmentY, alignmentA});
                    }
                    for (int N = 7; N <= (1 << 10); N *= 7) {
                      int lda = N + lda_diff;
                      b->Args({N, order, Uplo, alpha, incX, incY, lda,
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

BENCHMARK(BM_cblas_dsyr2)
    ->ArgNames({"N", "order", "Uplo", "alpha", "incX", "incY", "lda",
                "alignmentX", "alignmentY", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_dsyr2)
    ->ArgNames({"N", "order", "Uplo", "alpha", "incX", "incY", "lda",
                "alignmentX", "alignmentY", "alignmentA"})
    ->Apply(CustomArgumentsPacked);
