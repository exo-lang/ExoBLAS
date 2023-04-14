#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_idamax.h"
#include "generate_buffer.h"

static void BM_cblas_idamax(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  size_t alignmentX = state.range(2);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);

  for (auto _ : state) {
    cblas_idamax(N, X.data(), incX);
  }
}

static void BM_exo_idamax(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  size_t alignmentX = state.range(2);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);

  for (auto _ : state) {
    exo_idamax(N, X.data(), incX);
  }
}

BENCHMARK(BM_cblas_idamax)
    ->ArgNames({"n", "incX", "alignmentX"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2), {1}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {64}});
BENCHMARK(BM_exo_idamax)
    ->ArgNames({"n", "incX", "alignmentX"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2), {1}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7), {1}, {64}});

// BENCHMARK(BM_cblas_idamax)->ArgNames({"n", "incX",
// "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10},
//       {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4,
//       10}, {64}
//     });
// BENCHMARK(BM_exo_idamax)->ArgNames({"n", "incX",
// "alignmentX"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10},
//       {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4,
//       10}, {64}
//     });
