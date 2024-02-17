#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_rot_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(rot);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  int incY = state.range(2);
  T c = state.range(3);
  T s = state.range(4);
  size_t alignmentX = state.range(5);
  size_t alignmentY = state.range(6);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);

  for (auto _ : state) {
    rot<lib, T>(N, X.data(), incX, Y.data(), incY, c, s);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct({Ns,
                           {1},
                           {1},
                           {15},
                           {13},
                           {64},
                           {64},
                           {BENCH_TYPES::level_1},
                           {type_bits<T>()}});
  };
  b->ArgNames({"N", "incX", "incY", "c", "s", "alignmentX", "alignmentY",
               "bench_type", "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_7);
}

call_bench_all(rot);
