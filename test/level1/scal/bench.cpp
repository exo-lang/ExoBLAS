#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_scal_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(scal);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  T alpha = state.range(1);
  int incX = state.range(2);
  size_t alignmentX = state.range(3);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);

  for (auto _ : state) {
    scal<lib, T>(N, alpha, X.data(), incX);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct(
        {Ns, {15}, {1}, {64}, {BENCH_TYPES::level_1}, {type_bits<T>()}});
  };
  b->ArgNames({"N", "alpha", "incX", "alignmentX", "bench_type", "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_3);
}

call_bench_all(scal);
