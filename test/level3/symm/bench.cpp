#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_symm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(symm);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(1);
  const enum CBLAS_ORDER Order = (const enum CBLAS_ORDER)state.range(2);
  const enum CBLAS_SIDE Side = (const enum CBLAS_SIDE)state.range(3);
  const enum CBLAS_UPLO Uplo = (const enum CBLAS_UPLO)state.range(4);
  const T alpha = state.range(5);
  const int lda_diff = state.range(6);
  const int ldb = N + state.range(7);
  const T beta = state.range(8);
  const int ldc = N + state.range(9);
  const int alignmentA = state.range(10);
  const int alignmentB = state.range(11);
  const int alignmentC = state.range(12);

  int lda = Side == CBLAS_SIDE::CblasLeft ? M + lda_diff : N + lda_diff;
  int ka = Side == CBLAS_SIDE::CblasLeft ? M : N;

  auto A = AlignedBuffer2D<T>(ka, lda, alignmentA);
  auto B = AlignedBuffer2D<T>(M, ldb, alignmentB);
  auto C = AlignedBuffer2D<T>(M, ldc, alignmentC);

  for (auto _ : state) {
    symm<lib, T>(Order, Side, Uplo, M, N, alpha, A.data(), lda, B.data(), ldb,
                 beta, C.data(), ldc);
  }
}

template <typename T, int order, int Side, int Uplo, int TransA, int TransB,
          int Diag>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int M, int N) {
    return b->Args({M,
                    N,
                    order,
                    Side,
                    Uplo,
                    17,
                    0,
                    0,
                    18,
                    0,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_3_eq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"M", "N", "Order", "Side", "Uplo", "alpha", "lda_diff",
               "ldb_diff", "beta", "ldc_diff", "alignmentA", "alignmentB",
               "alignmentC", "bench_type", "precision"});
  for (int i = 1; i <= level_3_max_N; i *= 2) {
    add_arg(i, i);
  }
  for (int i = 3; i <= level_3_max_N; i *= 3) {
    add_arg(i, i);
  }
}

call_bench_all(symm, CblasRowMajor, CblasLeft, CblasLower, 0, 0, 0);
call_bench_all(symm, CblasRowMajor, CblasLeft, CblasUpper, 0, 0, 0);
