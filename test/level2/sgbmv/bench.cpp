#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_sgbmv.h"
#include "generate_buffer.h"

static void BM_cblas_sgbmv(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  int KL = state.range(1) == 1 ? 1 : M / 4;
  int KU = state.range(1) == 1 ? 1 : N / 4;
  float alpha = state.range(2);
  float beta = state.range(3);
  int incX = 1;
  int incY = 1;
  int lda = KL + KU + 1;

  auto X = AlignedBuffer<float>(M, incX, 64);
  auto Y = AlignedBuffer<float>(N, incY, 64);
  auto A = AlignedBuffer<float>(M * lda, 1, 64);

  for (auto _ : state) {
    cblas_sgbmv(CblasRowMajor, CblasNoTrans, M, N, KL, KU, alpha, A.data(), lda,
                X.data(), incX, beta, Y.data(), incY);
  }
}

static void BM_exo_sgbmv(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  int KL = state.range(1) == 1 ? 1 : M / 4;
  int KU = state.range(1) == 1 ? 1 : N / 4;
  float alpha = state.range(2);
  float beta = state.range(3);
  int incX = 1;
  int incY = 1;
  int lda = KL + KU + 1;

  auto X = AlignedBuffer<float>(M, incX, 64);
  auto Y = AlignedBuffer<float>(N, incY, 64);
  auto A = AlignedBuffer<float>(M * lda, 1, 64);

  for (auto _ : state) {
    exo_sgbmv(M, N, KL, KU, alpha, A.data(), lda, X.data(), incX, beta,
              Y.data(), incY);
  }
}

BENCHMARK(BM_cblas_sgbmv)
    ->ArgNames({"m", "n", "kl", "ku", "alpha", "beta", "incX", "incY"})
    ->ArgsProduct(
        {benchmark::CreateRange(8, (1 << 13), 2), {1, 2}, {0, 1, 3}, {0, 1, 3}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 13) - 1, 7),
                   {1, 2},
                   {0, 1, 3},
                   {0, 1, 3}});

BENCHMARK(BM_exo_sgbmv)
    ->ArgNames({"m", "kl", "alpha", "beta"})
    ->ArgsProduct(
        {benchmark::CreateRange(8, (1 << 13), 2), {1, 2}, {0, 1, 3}, {0, 1, 3}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 13) - 1, 7),
                   {1, 2},
                   {0, 1, 3},
                   {0, 1, 3}});
