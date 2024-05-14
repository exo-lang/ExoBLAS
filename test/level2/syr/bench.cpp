#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_syr_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  const int N = state.range(0);
  const enum CBLAS_ORDER order = state.range(1) == CBLAS_ORDER::CblasRowMajor
                                     ? CBLAS_ORDER::CblasRowMajor
                                     : CBLAS_ORDER::CblasColMajor;
  const enum CBLAS_UPLO Uplo = state.range(2) == CBLAS_UPLO::CblasUpper
                                   ? CBLAS_UPLO::CblasUpper
                                   : CBLAS_UPLO::CblasLower;
  const T alpha = state.range(3);
  const int lda = N + state.range(4);
  const int incX = state.range(5);
  size_t alignmentA = state.range(6);
  size_t alignmentX = state.range(7);

  auto A = AlignedBuffer2D<T>(N, lda, alignmentA);
  auto X = AlignedBuffer<T>(N, incX, alignmentX);

  for (auto _ : state) {
    syr<lib, T>(order, Uplo, N, alpha, X.data(), incX, A.data(), lda);
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
                    64,
                    64,
                    {BENCH_TYPES::level_2_sq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"N", "order", "Uplo", "alpha", "lda_diff", "incX", "alignmentA",
               "alignmentX", "bench_type", "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i);
  }
  for (int i = 3; i <= level_2_max_N; i *= 3) {
    add_arg(i);
  }
}

call_bench_all(syr, CblasRowMajor, CblasUpper, 0, 0);
call_bench_all(syr, CblasRowMajor, CblasLower, 0, 0);
