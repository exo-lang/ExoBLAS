#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_sgemv.h"
#include "generate_buffer.h"

static void BM_cblas_sgemv(benchmark::State &state) {
  const int N = state.range(0);
  const int M = N;
  const enum CBLAS_ORDER order = state.range(1) == 0
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_TRANSPOSE TransA = state.range(2) == 0
                                          ? CBLAS_TRANSPOSE::CblasNoTrans
                                          : CBLAS_TRANSPOSE::CblasTrans;
  const float alpha = state.range(3);
  const int lda = state.range(4) + N;
  const int incX = state.range(5);
  const float beta = state.range(6);
  const int incY = state.range(7);
  size_t alignmentA = state.range(8);
  size_t alignmentX = state.range(9);
  size_t alignmentY = state.range(10);

  auto A = AlignedBuffer<float>(N * lda, 1, alignmentA);
  int sizeX = N;
  int sizeY = M;
  if (TransA == CBLAS_TRANSPOSE::CblasTrans) {
    sizeX = M;
    sizeY = N;
  }

  auto X = AlignedBuffer<float>(sizeX, incX);
  auto Y = AlignedBuffer<float>(sizeY, incY);

  for (auto _ : state) {
    cblas_sgemv(order, TransA, N, N, alpha, A.data(), lda, X.data(), incX, beta,
                Y.data(), incY);
  }
}

static void BM_exo_sgemv(benchmark::State &state) {
  const int N = state.range(0);
  const int M = N;
  const enum CBLAS_ORDER order = state.range(1) == 0
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_TRANSPOSE TransA = state.range(2) == 0
                                          ? CBLAS_TRANSPOSE::CblasNoTrans
                                          : CBLAS_TRANSPOSE::CblasTrans;
  const float alpha = state.range(3);
  const int lda = state.range(4) + N;
  const int incX = state.range(5);
  const float beta = state.range(6);
  const int incY = state.range(7);
  size_t alignmentA = state.range(8);
  size_t alignmentX = state.range(9);
  size_t alignmentY = state.range(10);

  auto A = AlignedBuffer<float>(M * lda, 1, alignmentA);
  int sizeX = N;
  int sizeY = M;
  if (TransA == CBLAS_TRANSPOSE::CblasTrans) {
    sizeX = M;
    sizeY = N;
  }

  auto X = AlignedBuffer<float>(sizeX, incX);
  auto Y = AlignedBuffer<float>(sizeY, incY);

  for (auto _ : state) {
    exo_sgemv(order, TransA, N, N, alpha, A.data(), lda, X.data(), incX, beta,
              Y.data(), incY);
  }
}

static void CustomArgumentsPacked(benchmark::internal::Benchmark *b) {
  for (int order = 0; order < 1; ++order) {
    for (int TransA = 0; TransA <= 1; ++TransA) {
      for (int alpha = 3; alpha <= 3; ++alpha) {
        for (int lda_diff = 0; lda_diff <= 0; ++lda_diff) {
          for (int incX = 1; incX <= 1; ++incX) {
            for (int beta = 5; beta <= 5; ++beta) {
              for (int incY = 1; incY <= 1; ++incY) {
                for (int alignmentA = 64; alignmentA <= 64; ++alignmentA) {
                  for (int alignmentX = 64; alignmentX <= 64; ++alignmentX) {
                    for (int alignmentY = 64; alignmentY <= 64; ++alignmentY) {
                      for (int N = 1; N <= (1 << 13); N *= 2) {
                        b->Args({N, order, TransA, alpha, lda_diff, incX, beta,
                                 incY, alignmentA, alignmentX, alignmentY});
                      }
                      for (int N = 1; N <= (1 << 13); N *= 7) {
                        b->Args({N, order, TransA, alpha, lda_diff, incX, beta,
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

BENCHMARK(BM_cblas_sgemv)
    ->ArgNames({"n", "order", "TransA", "alpha", "lda_diff", "incX", "beta",
                "incY", "alignmentA", "alignmentX", "alignmentY"})
    ->Apply(CustomArgumentsPacked);
BENCHMARK(BM_exo_sgemv)
    ->ArgNames({"n", "order", "TransA", "alpha", "lda_diff", "incX", "beta",
                "incY", "alignmentA", "alignmentX", "alignmentY"})
    ->Apply(CustomArgumentsPacked);
