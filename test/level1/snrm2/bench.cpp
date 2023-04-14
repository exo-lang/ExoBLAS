#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_snrm2.h"
#include "generate_buffer.h"

static void BM_cblas_snrm2(benchmark::State &state) {
  auto N = state.range(0);
  auto incX = state.range(1);
  size_t alignmentX = state.range(2);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);

  for (auto _ : state) {
    cblas_snrm2(N, X.data(), incX);
  }
}

static void BM_exo_snrm2(benchmark::State &state) {
  auto N = state.range(0);
  auto incX = state.range(1);
  size_t alignmentX = state.range(2);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);

  for (auto _ : state) {
    exo_snrm2(N, X.data(), incX);
  }
}

BENCHMARK(BM_cblas_snrm2)
    ->ArgNames({"n", "incX", "alignmentX"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2), {1}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {64}});
BENCHMARK(BM_exo_snrm2)
    ->ArgNames({"n", "incX", "alignmentX"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2), {1}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {64}});

// BENCHMARK(BM_cblas_snrm2)->ArgNames({"n", "incX",
// "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
// BENCHMARK(BM_exo_snrm2)->ArgNames({"n", "incX", "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4, 10}
//     });
