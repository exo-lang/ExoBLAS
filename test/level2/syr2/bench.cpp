#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_syr2_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr2);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  CBLAS_ORDER order = state.range(1) == CBLAS_ORDER::CblasRowMajor
                          ? CBLAS_ORDER::CblasRowMajor
                          : CBLAS_ORDER::CblasColMajor;
  CBLAS_UPLO Uplo = state.range(2) == CBLAS_UPLO::CblasUpper
                        ? CBLAS_UPLO::CblasUpper
                        : CBLAS_UPLO::CblasLower;
  T alpha = state.range(3);
  int lda = N + state.range(4);
  int incX = state.range(5);
  int incY = state.range(6);
  size_t alignmentX = state.range(7);
  size_t alignmentY = state.range(8);
  size_t alignmentA = state.range(9);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);
  auto A = AlignedBuffer2D<T>(N, lda, alignmentA);

  for (auto _ : state) {
    syr2<lib, T>(order, Uplo, N, alpha, X.data(), incX, Y.data(), incY,
                 A.data(), lda);
  }
}

template <typename T, int order, int Uplo, int TransA, int Diag>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int N) {
    return b->Args({N,
                    order,
                    Uplo,
                    17,
                    0,
                    1,
                    1,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_2_sq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"N", "order", "Uplo", "alpha", "lda_diff", "incX", "incY",
               "alignmentA", "alignmentX", "alignmentY", "bench_type",
               "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i);
  }
  for (int i = 7; i <= level_2_max_N; i *= 7) {
    add_arg(i);
  }
}

call_bench_all(syr2, CblasRowMajor, CblasUpper, 0, 0);
call_bench_all(syr2, CblasRowMajor, CblasLower, 0, 0);
