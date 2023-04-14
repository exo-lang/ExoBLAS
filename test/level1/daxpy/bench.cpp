#include <benchmark/benchmark.h>
#include <cblas.h>

#include <vector>

#include "exo_daxpy.h"
#include "generate_buffer.h"

static void BM_cblas_daxpy(benchmark::State &state) {
  int N = state.range(0);
  double alpha = state.range(1);
  int incX = state.range(2);
  int incY = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);
  auto Y = AlignedBuffer<double>(N, incY, alignmentY);

  for (auto _ : state) {
    cblas_daxpy(N, alpha, X.data(), incY, Y.data(), incY);
  }
}

static void BM_exo_daxpy(benchmark::State &state) {
  int N = state.range(0);
  double alpha = state.range(1);
  int incX = state.range(2);
  int incY = state.range(3);
  size_t alignmentX = state.range(4);
  size_t alignmentY = state.range(5);

  auto X = AlignedBuffer<double>(N, incX, alignmentX);
  auto Y = AlignedBuffer<double>(N, incY, alignmentY);

  for (auto _ : state) {
    exo_daxpy(N, alpha, X.data(), incY, Y.data(), incY);
  }
}

BENCHMARK(BM_cblas_daxpy)
    ->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2),
                   {0, 1, 3},
                   {1},
                   {1},
                   {64},
                   {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {0, 1, 3},
                   {1},
                   {1},
                   {64},
                   {64}});
BENCHMARK(BM_exo_daxpy)
    ->ArgNames({"n", "alpha", "incX", "incY", "alignmentX", "alignmentY"})
    ->ArgsProduct({benchmark::CreateRange(1, (1 << 26), 2),
                   {0, 1, 3},
                   {1},
                   {1},
                   {64},
                   {64}})
    ->ArgsProduct({benchmark::CreateRange(7, (1 << 26) - 1, 7),
                   {0, 1, 3},
                   {1},
                   {1},
                   {64},
                   {64}});

// BENCHMARK(BM_cblas_daxpy)->ArgNames({"n", "alpha", "incX", "incY",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {3}, {{-4, 2, 4,
//       10}}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {3}, {-4, 2,
//       4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
// BENCHMARK(BM_exo_daxpy)->ArgNames({"n", "alpha", "incX", "incY",
// "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange((1 << 4), (1 << 24), (1 << 4)), {3}, {-4, 2, 4,
//       10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange((1 << 4) + 1, (1 << 24) - 1, 13), {3}, {-4, 2,
//       4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     });
