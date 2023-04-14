#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_sger.h"
#include "generate_buffer.h"

static void BM_cblas_sger(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  float alpha = state.range(1);
  int incX = 1;
  int incY = 1;
  int lda = N;
  size_t alignmentX = 64;
  size_t alignmentY = 64;
  size_t alignmentA = 64;

  auto X = AlignedBuffer<float>(M, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);
  auto A = AlignedBuffer<float>(M * lda, 1, alignmentA);

  for (auto _ : state) {
    cblas_sger(CBLAS_ORDER::CblasRowMajor, M, N, alpha, X.data(), incX,
               Y.data(), incY, A.data(), lda);
  }
}

static void BM_exo_sger(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  float alpha = state.range(1);
  int incX = 1;
  int incY = 1;
  int lda = N;
  size_t alignmentX = 64;
  size_t alignmentY = 64;
  size_t alignmentA = 64;

  auto X = AlignedBuffer<float>(M, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);
  auto A = AlignedBuffer<float>(M * lda, 1, alignmentA);

  for (auto _ : state) {
    exo_sger(M, N, alpha, X.data(), incX, Y.data(), incY, A.data(), lda);
  }
}

BENCHMARK(BM_cblas_sger)
    ->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY",
                "alignmentA"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 13), 2),
                   {1, 3}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 13) - 1, 7),
                   {1, 3}});
BENCHMARK(BM_exo_sger)
    ->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY",
                "alignmentA"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 13), 2),
                   {1, 3}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 13) - 1, 7),
                   {1, 3},});
