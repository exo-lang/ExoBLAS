#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_srot.h"
#include "generate_buffer.h"

static void BM_cblas_srot(benchmark::State &state) {
  auto N = state.range(0);
  auto incX = state.range(1);
  auto incY = state.range(2);
  float c = state.range(3);
  float s = state.range(4);
  size_t alignmentX = state.range(5);
  size_t alignmentY = state.range(6);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    cblas_srot(N, X.data(), incX, Y.data(), incY, c, s);
  }
}

static void BM_exo_srot(benchmark::State &state) {
  auto N = state.range(0);
  auto incX = state.range(1);
  auto incY = state.range(2);
  float c = state.range(3);
  float s = state.range(4);
  size_t alignmentX = state.range(5);
  size_t alignmentY = state.range(6);

  auto X = AlignedBuffer<float>(N, incX, alignmentX);
  auto Y = AlignedBuffer<float>(N, incY, alignmentY);

  for (auto _ : state) {
    exo_srot(N, X.data(), incX, Y.data(), incY, c, s);
  }
}

BENCHMARK(BM_cblas_srot)
    ->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2),
                   {1},
                   {1},
                   {2},
                   {3},
                   {64},
                   {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {1},
                   {1},
                   {2},
                   {3},
                   {64},
                   {64}});
BENCHMARK(BM_exo_srot)
    ->ArgNames({"n", "incX", "incY", "c", "s", "alignmentX", "alignmentY"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2),
                   {1},
                   {1},
                   {2},
                   {3},
                   {64},
                   {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {1},
                   {1},
                   {2},
                   {3},
                   {64},
                   {64}});

// BENCHMARK(BM_cblas_srot)->ArgNames({"n", "incX", "incY", "c", "s",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2), {-4, 2, 4, 10}, {-4, 2, 1, 4,
//       10}, {2}, {3}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7), {-4, 2, 4, 10}, {-4, 2, 1,
//       4, 10}, {2}, {3}, {64}, {64}
//     });
// BENCHMARK(BM_exo_srot)->ArgNames({"n", "incX", "incY", "c", "s",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2), {-4, 2, 4, 10}, {-4, 2, 1, 4,
//       10}, {2}, {3}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7), {-4, 2, 4, 10}, {-4, 2, 1,
//       4, 10}, {2}, {3}, {64}, {64}
//     });
