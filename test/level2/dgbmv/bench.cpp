#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_dgbmv.h"
#include "generate_buffer.h"

static void BM_cblas_dgbmv(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  int KL = state.range(1) == 1 ? M / 2 : M / 4;
  int KU = state.range(1) == 1 ? N / 2 : N / 4;
  double alpha = state.range(2);
  double beta = state.range(3);
  int incX = 1;
  int incY = 1;
  int lda = KL + KU + 1;

  auto X = AlignedBuffer<double>(N, incX, 64);
  auto Y = AlignedBuffer<double>(M, incY, 64);
  auto A = AlignedBuffer<double>(M * lda, 1, 64);

  for (auto _ : state) {
    cblas_dgbmv(CblasRowMajor, CblasNoTrans, M, N, KL, KU, alpha, A.data(), lda,
                X.data(), incX, beta, Y.data(), incY);
  }
}

static void BM_exo_dgbmv(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(0);
  int KL = state.range(1) == 1 ? M / 2 : M / 4;
  int KU = state.range(1) == 1 ? N / 2 : N / 4;
  double alpha = state.range(2);
  double beta = state.range(3);
  int incX = 1;
  int incY = 1;
  int lda = KL + KU + 1;

  auto X = AlignedBuffer<double>(N, incX, 64);
  auto Y = AlignedBuffer<double>(M, incY, 64);
  auto A = AlignedBuffer<double>(M * lda, 1, 64);

  for (auto _ : state) {
    exo_dgbmv(M, N, KL, KU, alpha, A.data(), lda, X.data(), incX, beta,
              Y.data(), incY);
  }
}

// CBLAS dgbmv segfaults on M=N=(1 << 19) and KL=KU=1...
// Is it possible they just resort to GEMV's routine?
BENCHMARK(BM_cblas_dgbmv)
    ->ArgNames({"m", "kl_setting", "alpha", "beta"})
    ->ArgsProduct({benchmark::CreateRange(8, (1 << 13), 2), {1, 2}, {3}, {3}})
    ->ArgsProduct(
        {benchmark::CreateRange(7, (1 << 13) - 1, 7), {1, 2}, {3}, {3}});

BENCHMARK(BM_exo_dgbmv)
    ->ArgNames({"m", "kl_setting", "alpha", "beta"})
    ->ArgsProduct({benchmark::CreateRange(8, (1 << 13), 2), {1, 2}, {3}, {3}})
    ->ArgsProduct(
        {benchmark::CreateRange(7, (1 << 13) - 1, 7), {1, 2}, {3}, {3}});
