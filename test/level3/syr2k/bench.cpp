#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_syr2k_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(syr2k);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  int K = state.range(1);
  const enum CBLAS_ORDER Order = (const enum CBLAS_ORDER)state.range(2);
  const enum CBLAS_UPLO Uplo = (const enum CBLAS_UPLO)state.range(3);
  const enum CBLAS_TRANSPOSE Trans = (const enum CBLAS_TRANSPOSE)state.range(4);
  const T alpha = state.range(5);
  const int lda_diff = state.range(6);
  const int ldb_diff = state.range(7);
  const T beta = state.range(8);
  const int ldc = N + state.range(9);
  const int alignmentA = state.range(10);
  const int alignmentB = state.range(11);
  const int alignmentC = state.range(12);

  auto A_dims = get_dims(Trans, N, K, lda_diff);
  const int lda = A_dims.second;
  auto A = AlignedBuffer2D<T>(A_dims.first, A_dims.second, alignmentA);
  auto B_dims = get_dims(Trans, N, K, ldb_diff);
  const int ldb = B_dims.second;
  auto B = AlignedBuffer2D<T>(B_dims.first, B_dims.second, alignmentB);
  auto C = AlignedBuffer2D<T>(N, ldc, alignmentC);

  for (auto _ : state) {
    syr2k<lib, T>(Order, Uplo, Trans, N, K, alpha, A.data(), lda, B.data(), ldb,
                  beta, C.data(), ldc);
  }
}

template <typename T, int order, int Uplo, int Trans>
static void args(benchmark::internal::Benchmark *b) {
  auto add_arg = [&b](int N, int K) {
    return b->Args({N,
                    K,
                    order,
                    Uplo,
                    Trans,
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
  b->ArgNames({"N", "K", "order", "Uplo", "Trans", "alpha", "lda_diff",
               "ldb_diff", "beta", "ldc_diff", "alignmentA", "alignmentB",
               "alignmentC", "bench_type", "precision"});
  for (int i = 1; i <= level_3_max_N; i *= 2) {
    add_arg(i, i);
  }
  for (int i = 7; i <= level_3_max_N; i *= 7) {
    add_arg(i, i);
  }
}

#define call_syr2k_bench(lib, T, order, Uplo, Trans)                         \
  BENCHMARK(bench<lib, T>)                                                   \
      ->Name(level_3_kernel_name<lib, T>("syr2k", order, 0, Uplo, Trans, 0)) \
      ->Apply(args<T, order, Uplo, Trans>);

#define call_syr2k_bench_all(order, Uplo, Trans)      \
  call_syr2k_bench(Exo, float, order, Uplo, Trans);   \
  call_syr2k_bench(Cblas, float, order, Uplo, Trans); \
  call_syr2k_bench(Exo, double, order, Uplo, Trans);  \
  call_syr2k_bench(Cblas, double, order, Uplo, Trans);

call_syr2k_bench_all(CBLAS_ORDER::CblasRowMajor, CBLAS_UPLO::CblasLower,
                     CBLAS_TRANSPOSE::CblasNoTrans);
