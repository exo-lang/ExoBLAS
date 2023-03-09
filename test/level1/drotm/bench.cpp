#include <vector>

#include <cblas.h>
#include <benchmark/benchmark.h>

#include "generate_buffer.h"

#include "exo_drotm.h"

static void BM_cblas_drotm(benchmark::State& state) {
    int N = state.range(0);
    double HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);
    double H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

    for (auto _ : state) {
        cblas_drotm(N, X.data(), incX, Y.data(), incY, H);
    }
}

static void BM_exo_drotm(benchmark::State& state) {
    int N = state.range(0);
    double HFlag = state.range(1);
    int incX = state.range(2);
    int incY = state.range(3);
    size_t alignmentX = state.range(4);
    size_t alignmentY = state.range(5);

    auto X = AlignedBuffer<double>(N, incX, alignmentX);
    auto Y = AlignedBuffer<double>(N, incY, alignmentY);
    double H[5] = {HFlag, 1.2, 2.2, 3.2, 4.2};

    for (auto _ : state) {
        exo_drotm(N, X.data(), incX, Y.data(), incY, H);
    }
}

BENCHMARK(BM_cblas_drotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    });
BENCHMARK(BM_exo_drotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
      benchmark::CreateRange(1, (1 << 26), 2),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    })->ArgsProduct({
      benchmark::CreateRange(7, (1 << 26) - 1, 7),
      {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
    });

// BENCHMARK(BM_cblas_drotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2),
//       {-1, 0, 1, -2}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//     benchmark::CreateRange(7, (1 << 26) - 1, 7),
//     {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
//     });
// BENCHMARK(BM_exo_drotm)->ArgNames({"n", "HFlag", "incX", "incY", "alignmentX", "alignmentY"})->ArgsProduct({
//       benchmark::CreateRange(1, (1 << 26), 2),
//       {-1, 0, 1, -2}, {-4, 2, 4, 10}, {-4, 2, 1, 4, 10}, {64}, {64}
//     })->ArgsProduct({
//       benchmark::CreateRange(7, (1 << 26) - 1, 7),
//       {-1, 0, 1, -2}, {1}, {1}, {64}, {64}
//     });