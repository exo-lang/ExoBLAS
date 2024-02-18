#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_rotm_wrapper.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(rotm);

template <typename lib, typename T>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  T Flag = state.range(1);
  int incX = state.range(2);
  int incY = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<T>(N, incX, alignmentX);
  auto Y = AlignedBuffer<T>(N, incY, alignmentY);
  T H[5] = {Flag, 1.2, 2.2, 3.2, 4.2};

  for (auto _ : state) {
    rotm<lib, T>(N, X.data(), incX, Y.data(), incY, H);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct({Ns,
                           {-1, 0, 1, -2},
                           {1},
                           {1},
                           {64},
                           {64},
                           {BENCH_TYPES::level_1},
                           {type_bits<T>()}});
  };
  b->ArgNames({"N", "Flag", "incX", "incY", "alignmentX", "alignmentY",
               "bench_type", "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_7);
}

call_bench_all(rotm);
