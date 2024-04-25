#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_gemv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(gemv);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  const int M = state.range(0);
  const int N = state.range(1);
  const enum CBLAS_ORDER Order = (const enum CBLAS_ORDER)state.range(2);
  const enum CBLAS_TRANSPOSE TransA =
      (const enum CBLAS_TRANSPOSE)state.range(3);
  const T alpha = state.range(4);
  const int lda = N + state.range(5);
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

  auto X = AlignedBuffer<T>(sizeX, incX, alignmentX);
  auto Y = AlignedBuffer<T>(sizeY, incY, alignmentY);

  for (auto _ : state) {
    gemv<lib, T>(Order, TransA, M, N, alpha, A.data(), lda, X.data(), incX,
                 beta, Y.data(), incY);
  }
}

template <typename T, int Order, int Uplo, int TransA, int Diag>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int M, int N) {
    return b->Args({M,
                    N,
                    Order,
                    TransA,
                    17,
                    0,
                    1,
                    14,
                    1,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_2_sq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"M", "N", "Order", "TransA", "alpha", "lda_diff", "incX", "beta",
               "incY", "alignmentA", "alignmentX", "alignmentY", "bench_type",
               "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i, 50);
  }
}

call_bench_all(gemv, CblasRowMajor, 0, CblasNoTrans, 0);
call_bench_all(gemv, CblasRowMajor, 0, CblasTrans, 0);
