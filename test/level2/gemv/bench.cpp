#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_gemv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(gemv);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  const int N = state.range(0);
  const int M = state.range(1);
  const enum CBLAS_ORDER order = state.range(2) == CBLAS_ORDER::CblasRowMajor
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_TRANSPOSE TransA =
      state.range(3) == CBLAS_TRANSPOSE::CblasNoTrans
          ? CBLAS_TRANSPOSE::CblasNoTrans
          : CBLAS_TRANSPOSE::CblasTrans;
  const T alpha = state.range(4);
  const int lda = M + state.range(5);
  const int incX = state.range(6);
  const T beta = state.range(7);
  const int incY = state.range(8);
  size_t alignmentA = state.range(9);
  size_t alignmentX = state.range(10);
  size_t alignmentY = state.range(11);

  auto A = AlignedBuffer2D<T>(M, lda, alignmentA);
  int sizeX = N;
  int sizeY = M;
  if (TransA == CBLAS_TRANSPOSE::CblasTrans) {
    sizeX = M;
    sizeY = N;
  }

  auto X = AlignedBuffer<T>(sizeX, incX);
  auto Y = AlignedBuffer<T>(sizeY, incY);

  for (auto _ : state) {
    gemv<lib, T>(order, TransA, N, N, alpha, A.data(), lda, X.data(), incX,
                 beta, Y.data(), incY);
  }
}

template <typename T, int order, int TransA, int Uplo>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int M, int N) {
    return b->Args({M,
                    N,
                    order,
                    TransA,
                    17,
                    0,
                    1,
                    14,
                    1,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_2_eq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"M", "N", "order", "TransA", "alpha", "lda_diff", "incX", "beta",
               "incY", "alignmentA", "alignmentX", "alignmentY", "bench_type",
               "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i, i);
  }
  for (int i = 7; i <= level_2_max_N; i *= 7) {
    add_arg(i, i);
  }
}

call_bench_all(gemv, CblasRowMajor, CblasNoTrans, 0);
call_bench_all(gemv, CblasRowMajor, CblasTrans, 0);
