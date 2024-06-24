#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_axpy_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(axpy);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  T alpha = state.range(1);
  int incX = state.range(2);
  int incY = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);

  for (auto _ : state) {
    axpy<lib, T>(N, alpha, X.data(), incY, Y.data(), incY);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct({Ns,
                           {15},
                           {1},
                           {1},
                           {64},
                           {64},
                           {BENCH_TYPES::level_1},
                           {type_bits<T>()}});
  };
  b->ArgNames({"N", "alpha", "incX", "incY", "alignmentX", "alignmentY",
               "bench_type", "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_3);
}

call_bench_all(axpy);
