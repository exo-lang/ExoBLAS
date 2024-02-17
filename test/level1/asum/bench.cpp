#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_asum_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper_ret(asum);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(1);
  int incX = state.range(2);
  size_t alignmentX = state.range(3);
  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  for (auto _ : state) {
    asum<lib, T>(N, X.data(), incX);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  b->ArgNames({"N", "incX", "alignmentX", "bench_type", "precision"})
      ->ArgsProduct(
          {level_1_pow_2, {1}, {64}, {BENCH_TYPES::level_1}, {type_bits<T>()}})
      ->ArgsProduct(
          {level_1_pow_7, {1}, {64}, {BENCH_TYPES::level_1}, {type_bits<T>()}});
}

call_bench_all(asum);
