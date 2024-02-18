#include <benchmark/benchmark.h>
#include <cblas.h>

#include "bench_ranges.h"
#include "exo_sdsdot.h"
#include "generate_buffer.h"
#include "misc.h"

generate_wrapper(dsdot);

template <typename lib>
static void bench(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  int incY = state.range(2);
  float alpha = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    dsdot<lib, float>(N, alpha, X.data(), incX, Y.data(), incY);
  }
}

template <typename T>
static void args(benchmark::internal::Benchmark *b) {
  auto add_args = [&b](auto Ns) {
    return b->ArgsProduct({Ns,
                           {1},
                           {1},
                           {4},
                           {64},
                           {64},
                           {BENCH_TYPES::level_1},
                           {type_bits<T>()}});
  };
  b->ArgNames({"N", "incX", "incY", "alpha", "alignmentX", "alignmentY",
               "bench_type", "precision"});
  add_args(level_1_pow_2);
  add_args(level_1_pow_7);
}

BENCHMARK(bench<Exo>)->Name("exo_sdsdot")->Apply(args<float>);
BENCHMARK(bench<Cblas>)->Name("cblas_sdsdot")->Apply(args<float>);
