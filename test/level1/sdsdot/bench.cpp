#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_sdsdot.h"
#include "generate_buffer.h"

static void BM_cblas_sdsdot(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  int incY = state.range(2);
  float alpha = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    cblas_sdsdot(N, alpha, X.data(), incX, Y.data(), incY);
  }
}

static void BM_exo_sdsdot(benchmark::State &state) {
  int N = state.range(0);
  int incX = state.range(1);
  int incY = state.range(2);
  float alpha = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    exo_sdsdot(N, alpha, X.data(), incX, Y.data(), incY);
  }
}

BENCHMARK(BM_cblas_sdsdot)
    ->ArgNames({"n", "incX", "incY", "alpha", "alignmentX", "alignmentY"})
    ->ArgsProduct(
        {benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {1}, {64}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {1},
                   {1},
                   {1},
                   {64},
                   {64}});
BENCHMARK(BM_exo_sdsdot)
    ->ArgNames({"n", "incX", "incY", "alpha", "alignmentX", "alignmentY"})
    ->ArgsProduct(
        {benchmark::CreateRange(1, (1 << 26), 2), {1}, {1}, {1}, {64}, {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {1},
                   {1},
                   {1},
                   {64},
                   {64}});

// BENCHMARK(BM_cblas_sdsdot)->ArgNames({"n", "incX", "incY", "alpha",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10},
//       {-4, 2, 1, 4, 10}, {1}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4,
//       10}, {-4, 2, 1, 4, 10}, {1}, {64}, {64}
//     });
// BENCHMARK(BM_exo_sdsdot)->ArgNames({"n", "incX", "incY", "alpha",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {-4, 2, 4, 10},
//       {-4, 2, 1, 4, 10}, {1}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {-4, 2, 4,
//       10}, {-4, 2, 1, 4, 10}, {1}, {64}, {64}
//     });
