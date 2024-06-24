#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_copy_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(copy);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  int incY = state.range(2);
  size_t alignmentX = state.range(3);
  size_t alignmentY = state.range(4);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);

  for (auto _ : state) {
    copy<lib, T>(N, X.data(), incX, Y.data(), incY);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct(
        {Ns, {1}, {1}, {64}, {64}, {BENCH_TYPES::level_1}, {type_bits<T>()}});
  };
  b->ArgNames({"N", "incX", "incY", "alignmentX", "alignmentY", "bench_type",
               "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_3);
}

call_bench_all(copy);
