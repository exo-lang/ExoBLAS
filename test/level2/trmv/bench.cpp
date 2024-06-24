#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_trmv_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(trmv);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  const int N = state.range(0);
  const enum CBLAS_ORDER order = (const enum CBLAS_ORDER)state.range(1);
  const enum CBLAS_UPLO Uplo = (const enum CBLAS_UPLO)state.range(2);
  const enum CBLAS_TRANSPOSE TransA =
      (const enum CBLAS_TRANSPOSE)state.range(3);
  const enum CBLAS_DIAG Diag = (const enum CBLAS_DIAG)state.range(4);
  const int lda = N + state.range(5);
  const int incX = state.range(6);
  size_t alignmentA = state.range(7);
  size_t alignmentX = state.range(8);

  auto A = AlignedBuffer2D<T>(N, lda, alignmentA);
  auto X = AlignedBuffer<T>(N, incX, alignmentX);

  for (auto _ : state) {
    trmv<lib, T>(order, Uplo, TransA, Diag, N, A.data(), lda, X.data(), incX);
  }
}

template <typename T, int order, int Uplo, int TransA, int Diag>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int N) {
    return b->Args({N,
                    order,
                    Uplo,
                    TransA,
                    Diag,
                    0,
                    1,
                    64,
                    64,
                    {BENCH_TYPES::level_2_sq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"N", "order", "Uplo", "TransA", "Diag", "lda_diff", "incX",
               "alignmentA", "alignmentX", "bench_type", "precision"});
  for (int i = 1; i <= level_2_max_N; i *= 2) {
    add_arg(i);
  }
  for (int i = 3; i <= level_2_max_N; i *= 3) {
    add_arg(i);
  }
}

call_bench_all(trmv, CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit);
call_bench_all(trmv, CblasRowMajor, CblasLower, CblasTrans, CblasNonUnit);
call_bench_all(trmv, CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit);
call_bench_all(trmv, CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit);
call_bench_all(trmv, CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit);
call_bench_all(trmv, CblasRowMajor, CblasLower, CblasTrans, CblasUnit);
call_bench_all(trmv, CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit);
call_bench_all(trmv, CblasRowMajor, CblasUpper, CblasTrans, CblasUnit);
