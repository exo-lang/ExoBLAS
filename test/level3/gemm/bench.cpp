#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_gemm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(gemm);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int M = state.range(0);
  int N = state.range(1);
  int K = state.range(2);
  const enum CBLAS_ORDER order = (const enum CBLAS_ORDER)state.range(3);
  const enum CBLAS_TRANSPOSE TransA =
      (const enum CBLAS_TRANSPOSE)state.range(4);
  const enum CBLAS_TRANSPOSE TransB =
      (const enum CBLAS_TRANSPOSE)state.range(5);
  const T alpha = state.range(6);
  const int lda_diff = state.range(7);
  const int ldb_diff = state.range(8);
  const T beta = state.range(9);
  const int ldc = N + state.range(10);
  const int alignmentA = state.range(11);
  const int alignmentB = state.range(12);
  const int alignmentC = state.range(13);

  auto A_dims = get_dims(TransA, M, K, lda_diff);
  const int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second, alignmentA);
  auto B_dims = get_dims(TransB, K, N, ldb_diff);
  const int ldb = B_dims.second;
  auto B = AlignedBuffer2D<T>(B_dims.first, B_dims.second, alignmentB);
  auto C = AlignedBuffer2D<T>(M, ldc, alignmentC);

  for (auto _ : state) {
    gemm<lib, T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                 A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
  }
}

template <typename T, int order, int Side, int Uplo, int TransA, int TransB>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int M, int N, int K) {
    return b->Args({M,
                    N,
                    K,
                    order,
                    TransA,
                    TransB,
                    17,
                    0,
                    0,
                    1,
                    0,
                    64,
                    64,
                    64,
                    {BENCH_TYPES::level_3_eq},
                    {type_bits<T>()}});
  };
  b->ArgNames({"M", "N", "K", "order", "TransA", "TransB", "alpha", "lda_diff",
               "ldb_diff", "beta", "ldc_diff", "alignmentA", "alignmentB",
               "alignmentC", "bench_type", "precision"});
  for (int i = 1; i <= level_3_max_N; i *= 2) {
    add_arg(i, i, i);
  }
  for (int i = 7; i <= level_3_max_N; i *= 7) {
    add_arg(i, i, i);
  }
}

call_bench_all(gemm, CblasRowMajor, 0, 0, CblasNoTrans, CblasNoTrans);
