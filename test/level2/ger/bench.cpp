#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_ger_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(ger);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const enum CBLAS_ORDER Order = (const enum CBLAS_ORDER)state.range(2);
  const T alpha = state.range(3);
  const int incX = state.range(4);
  const int incY = state.range(5);
  const int lda = N + state.range(6);
  size_t alignmentA = state.range(7);
  size_t alignmentX = state.range(8);
  size_t alignmentY = state.range(9);

  auto A = AlignedBuffer2D<T>(M, lda, alignmentA);
  auto X = AlignedBuffer<T>(M, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);

  for (auto _ : state) {
    ger<lib, T>(Order, M, N, alpha, X.data(), incX, Y.data(), incY, A.data(),
                lda);
  }
}

template <typename T, int Order, int Uplo, int TransA, int Diag>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int M, int N) {
    return b->Args({M,
                    N,
                    Order,
                    17,
                    1,
                    1,
                    0,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_2_sq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"M", "N", "Order", "alpha", "incX", "incY", "lda_diff",
               "alignmentA", "alignmentX", "alignmentY", "bench_type",
               "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i, i);
  }
  for (int i = 7; i <= level_2_max_N; i *= 7) {
    add_arg(i, i);
  }
}

call_bench_all(ger, CblasRowMajor, 0, 0, 0);
