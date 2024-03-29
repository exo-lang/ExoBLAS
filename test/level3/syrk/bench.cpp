#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_syrk_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syrk);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  int K = state.range(1);
  const enum CBLAS_ORDER Order = (const enum CBLAS_ORDER)state.range(2);
  const enum CBLAS_UPLO Uplo = (const enum CBLAS_UPLO)state.range(3);
  const enum CBLAS_TRANSPOSE Trans = (const enum CBLAS_TRANSPOSE)state.range(4);
  const T alpha = state.range(5);
  const int lda_diff = state.range(6);
  const T beta = state.range(7);
  const int ldc = N + state.range(8);
  const int alignmentA = state.range(9);
  const int alignmentC = state.range(10);

  auto A_dims = get_dims(Trans, N, K, lda_diff);
  const int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second, alignmentA);
  auto C = AlignedBuffer2D<T>(N, ldc, alignmentC);

  for (auto _ : state) {
    syrk<lib, T>(Order, Uplo, Trans, N, K, alpha, A.data(), lda, beta, C.data(),
                 ldc);
  }
}

template <typename T, int order, int Side, int Uplo, int TransA, int TransB>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int N, int K) {
    return b->Args({N,
                    K,
                    order,
                    Uplo,
                    TransA,
                    17,
                    0,
                    18,
                    0,
                    64,
                    64,
                    {BENCH_TYPES::level_3_eq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"N", "K", "order", "Uplo", "Trans", "alpha", "lda_diff", "beta",
               "ldc_diff", "alignmentA", "alignmentC", "bench_type",
               "precision"});
  for (int i = 1; i <= level_3_max_N; i *= 2) {
    add_arg(i, i);
  }
  for (int i = 7; i <= level_3_max_N; i *= 7) {
    add_arg(i, i);
  }
}

call_bench_all(syrk, CblasRowMajor, 0, CblasLower, CblasNoTrans, 0);
